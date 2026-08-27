# %% [markdown]
# # Digikala Recommendation Status — XLM-RoBERTa روی train ده‌درصدی
#
# این Notebook با بودجهٔ حدود **۱۰٪ داده** برای بهترکردن `Macro-F1` طراحی شده است. مدل پایه،
# preprocessing، ترتیب برچسب‌ها و test/validation نسخهٔ ۲٪ ثابت می‌مانند؛ train بزرگ‌تر،
# group-safe و از گروه‌های جدیدِ دارای label conflict پاک می‌شود. بنابراین این یک آزمایش بهینه‌سازی
# است، نه برآورد علّیِ خالص اثر حجم داده.
#
# اجرای کار دو فاز دارد:
#
# 1. `validation`: ساخت train ده‌درصدی، آموزش از checkpoint پایه و ارزیابی روی validation قفل‌شده.
# 2. `final`: فقط پس از PASS فاز اول، آموزش نهایی از همان checkpoint پایه و یک بار ارزیابی test.
#
# نسخهٔ عمومی ۲٪ هرگز overwrite نمی‌شود. نسخهٔ ۱۰٪ فقط در صورت عبور از همهٔ گیت‌ها نام v2 می‌گیرد.

# %% [markdown]
# ## ورودی‌های Kaggle و روش اجرا
#
# در `Settings > Accelerator` یک **GPU T4** و در `Settings > Internet` گزینهٔ On را انتخاب کنید.
# سپس با `Add Input` این دو خروجی را اضافه کنید:
#
# - خروجی baseline که فایل `sampled_split_manifest.csv` را دارد.
# - Dataset عمومی نسخهٔ فعلی:
#   `maslri/digikala-recommendation-status-xlm-roberta-v1`
#
# اجرای اول: `RUN_PHASE = 'validation'`. بعد از اتمام، Output را `Save Version` کنید.
# اگر validation PASS شد، همان Saved Version را به‌عنوان Input اضافه کنید، مقدار را به
# `RUN_PHASE = 'final'` تغییر دهید و دوباره اجرا کنید.
#
# checkpointهای داخل `/kaggle/working` فقط در همان session باقی می‌مانند. برای ادامه در session
# جدید باید Output اجرای نیمه‌تمام را Save Version و در اجرای جدید Add Input کنید.

# %%
from __future__ import annotations

import subprocess
import sys

# این سلول باید پیش از import کردن torch/transformers اجرا شود. torch پیش‌فرض Kaggle تغییر نمی‌کند.
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'transformers==4.57.6', 'accelerate>=1.2,<2',
    'sentencepiece>=0.2', 'safetensors>=0.4',
])
print('Dependencies are ready. If this cell changed an already-imported package, restart the session once.')

# %%
import gc
import hashlib
import inspect
import io
import json
import math
import os
import platform
import random
import shutil
import socket
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from IPython.display import display
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# فقط این مقدار را بین دو اجرای اصلی تغییر دهید: validation یا final
RUN_PHASE = os.getenv('DIGIKALA_10PCT_PHASE', 'validation').strip().lower()
if RUN_PHASE not in {'validation', 'final'}:
    raise ValueError("RUN_PHASE must be 'validation' or 'final'.")

SEED = 42
VALID_LABELS = ['recommended', 'not_recommended', 'no_idea']
LABEL2ID = {label: index for index, label in enumerate(VALID_LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

HF_REPO_ID = 'RadeAI/Digikala_comments_products'
HF_REVISION = '89c3133b169c8d3793db8834f56f32fee33d9db0'
HF_FILENAME = 'digikala-comments.csv'
HF_EXPECTED_SIZE = 1_278_526_959
HF_EXPECTED_SHA256 = 'c7a8aa3020334fde8ec24944576a03fe5785e6fe12cd01042f5836632ddf8297'
HF_DOWNLOAD_URL = f'https://huggingface.co/datasets/{HF_REPO_ID}/resolve/{HF_REVISION}/{HF_FILENAME}?download=true'

MODEL_NAME = 'FacebookAI/xlm-roberta-base'
MODEL_REVISION = 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'
MODEL_RC_VERSION = 'digikala-rec-xlm-roberta-10pct-v2.0.0-rc1'
MODEL_RELEASE_VERSION = 'digikala-rec-xlm-roberta-10pct-v2.0.0'
PREPROCESSING_VERSION = 'fa_light_v1'

TARGET_TRAIN_FRACTION = 0.10
CANDIDATE_POOL_FRACTION = 0.20
# بهینه‌سازی برای Macro-F1: enrichment ملایم کلاس‌های کم‌تعداد، نه balance کامل.
TARGET_TRAIN_CLASS_RATIOS = {
    'recommended': 0.70,
    'not_recommended': 0.14,
    'no_idea': 0.16,
}
if not math.isclose(sum(TARGET_TRAIN_CLASS_RATIOS.values()), 1.0, abs_tol=1e-12):
    raise ValueError('TARGET_TRAIN_CLASS_RATIOS must sum to 1.')
CHUNK_SIZE = 250_000
MAX_LENGTH = 128
EPOCHS = 2.0
TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EVAL_SAVE_STEPS = 2_000
BOOTSTRAP_ITERATIONS = 1_000

# Overrideهای اختیاری؛ معمولاً خالی بمانند.
COMMENTS_PATH_OVERRIDE = os.getenv('DIGIKALA_COMMENTS_PATH') or None
MANIFEST_PATH_OVERRIDE = os.getenv('DIGIKALA_MANIFEST_PATH') or None
RESUME_CHECKPOINT_OVERRIDE = os.getenv('DIGIKALA_RESUME_CHECKPOINT') or None
VALIDATION_SUMMARY_OVERRIDE = os.getenv('DIGIKALA_VALIDATION_SUMMARY_PATH') or None
V1_PREDICTIONS_OVERRIDE = os.getenv('DIGIKALA_V1_PREDICTIONS_PATH') or None

OUTPUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd() / 'outputs' / 'kaggle_10pct'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_RUN_DIR = OUTPUT_DIR / f'xlm_roberta_10pct_{RUN_PHASE}_run'
TRAIN_RUN_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_2PCT_PROFILE = {
    'train': {'rows': 85_694, 'text_groups': 66_899, 'recommended': 68_106, 'not_recommended': 8_159, 'no_idea': 9_429},
    'validation': {'rows': 9_941, 'text_groups': 8_249, 'recommended': 7_678, 'not_recommended': 993, 'no_idea': 1_270},
    'test': {'rows': 9_662, 'text_groups': 8_215, 'recommended': 7_612, 'not_recommended': 980, 'no_idea': 1_070},
}

V1_VALIDATION = {
    'macro_f1': 0.733375495566968,
    'f1_recommended': 0.9141566265060241,
    'f1_not_recommended': 0.7648780487804878,
    'f1_no_idea': 0.5210918114143921,
}
V1_TEST = {
    'macro_f1': 0.7171704486592764,
    'weighted_f1': 0.8553527913998584,
    'accuracy': 0.8422686814324156,
    'f1_recommended': 0.9218365673681334,
    'f1_not_recommended': 0.7594810379241517,
    'f1_no_idea': 0.470193740685544,
    'single_p95_ms_t4': 11.476199200015454,
    'batch_throughput_t4': 861.6425980886825,
    'peak_gpu_memory_bytes_t4': 1_414_517_248,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

# %% [markdown]
# ## بررسی سخت‌افزار
#
# این Notebook عمداً PyTorch را reinstall نمی‌کند. در نتیجه خطای قبلی P100 با build ناسازگار
# تکرار نمی‌شود؛ اگر GPU/build سازگار نباشند، همین ابتدا با پیام روشن متوقف می‌شویم.

# %%
if not torch.cuda.is_available():
    raise RuntimeError('GPU پیدا نشد. در Kaggle یک T4 انتخاب و session را restart کنید.')

GPU_NAME = torch.cuda.get_device_name(0)
GPU_CAPABILITY = tuple(torch.cuda.get_device_capability(0))
DEVICE_ARCH = f'sm_{GPU_CAPABILITY[0]}{GPU_CAPABILITY[1]}'
COMPILED_ARCHES = torch.cuda.get_arch_list()
if GPU_CAPABILITY < (7, 5) or DEVICE_ARCH not in COMPILED_ARCHES:
    raise RuntimeError(
        f'GPU/build ناسازگار است: gpu={GPU_NAME}, required={DEVICE_ARCH}, compiled={COMPILED_ARCHES}. '
        'برای این Notebook از T4 استفاده کنید.'
    )

# اجرای kernel واقعی، نه صرفاً cuda.is_available().
probe = (torch.ones((32, 32), device='cuda', dtype=torch.float16) @
         torch.ones((32, 32), device='cuda', dtype=torch.float16)).sum()
torch.cuda.synchronize()
del probe

runtime_info = {
    'phase': RUN_PHASE,
    'python': sys.version.split()[0],
    'torch': torch.__version__,
    'transformers': transformers.__version__,
    'scikit_learn': sklearn.__version__,
    'gpu': GPU_NAME,
    'gpu_compute_capability': list(GPU_CAPABILITY),
    'torch_cuda_runtime': torch.version.cuda,
    'compiled_cuda_arches': COMPILED_ARCHES,
    'precision': 'fp16',
}
print(json.dumps(runtime_info, ensure_ascii=False, indent=2))

# %% [markdown]
# ## دریافت منبع pin‌شده و manifest رسمی ۲٪

# %%
def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_source(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size != HF_EXPECTED_SIZE:
        raise ValueError(f'Unexpected source file or size: {path}')
    actual = file_sha256(path)
    if actual != HF_EXPECTED_SHA256:
        raise ValueError(f'Source SHA256 mismatch: {actual}')
    return path


def get_source_csv(override: str | None = None) -> Path:
    if override:
        return validate_source(Path(override))
    input_root = Path('/kaggle/input')
    if input_root.exists():
        for candidate in input_root.rglob(HF_FILENAME):
            if candidate.is_file() and candidate.stat().st_size == HF_EXPECTED_SIZE:
                try:
                    print('Validating attached source:', candidate)
                    return validate_source(candidate)
                except ValueError:
                    pass
    try:
        socket.getaddrinfo('huggingface.co', 443, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise RuntimeError('Kaggle Internet را روشن و session را restart کنید.') from error
    try:
        from huggingface_hub import hf_hub_download
        cached = Path(hf_hub_download(
            repo_id=HF_REPO_ID, repo_type='dataset', filename=HF_FILENAME,
            revision=HF_REVISION, cache_dir=str(OUTPUT_DIR / 'hf_cache'),
        ))
        return validate_source(cached)
    except Exception as first_error:
        print(f'huggingface_hub failed ({type(first_error).__name__}); using direct URL.')
        target = OUTPUT_DIR / 'hf_data' / HF_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix('.csv.part')
        request = urllib.request.Request(HF_DOWNLOAD_URL, headers={'User-Agent': 'digikala-10pct/2.0'})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
            downloaded = 0
            next_report = 128 * 1024 * 1024
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    print(f'Downloaded: {downloaded / 1_000_000_000:.2f} GB')
                    next_report += 128 * 1024 * 1024
        partial.replace(target)
        return validate_source(target)


def split_profile(frame: pd.DataFrame) -> dict:
    result = {}
    for split_name, group in frame.groupby('split', sort=False):
        result[str(split_name)] = {
            'rows': int(len(group)),
            'text_groups': int(group['text_group_id'].nunique()),
            **{label: int(group['recommendation_status'].eq(label).sum()) for label in VALID_LABELS},
        }
    return result


def load_official_manifest(override: str | None = None) -> tuple[pd.DataFrame, Path]:
    if override:
        candidates = [Path(override)]
    else:
        candidates = []
        for root in [Path('/kaggle/input'), Path('/kaggle/working'), Path.cwd()]:
            if root.exists():
                candidates.extend(root.rglob('sampled_split_manifest.csv'))
    errors = []
    valid = []
    ordered_columns = ['id', 'product_id', 'text_group_id', 'recommendation_status', 'split']
    for path in candidates:
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            needed = set(ordered_columns)
            if not needed.issubset(frame.columns):
                raise ValueError(f'missing columns: {needed - set(frame.columns)}')
            frame = frame[ordered_columns].copy()
            if split_profile(frame) != EXPECTED_2PCT_PROFILE:
                raise ValueError('profile mismatch')
            if frame['id'].duplicated().any():
                raise ValueError('duplicate id')
            lines = frame.astype(str).sort_values(ordered_columns).agg('\x1f'.join, axis=1)
            digest = hashlib.sha256('\n'.join(lines).encode('utf-8')).hexdigest()
            valid.append((frame, path, digest))
        except Exception as error:
            errors.append(f'{path}: {error}')
    if not valid:
        raise FileNotFoundError(
            'manifest رسمی ۲٪ پیدا نشد. Output baseline را Add Input کنید. Checked: ' + ' | '.join(errors[:5])
        )
    distinct_digests = {digest for _, _, digest in valid}
    if len(distinct_digests) > 1 and not override:
        raise RuntimeError(
            'چند manifest با profile یکسان ولی محتوای متفاوت پیدا شد. '
            'مسیر درست را در DIGIKALA_MANIFEST_PATH مشخص کنید.'
        )
    frame, path, _ = sorted(valid, key=lambda item: len(str(item[1])))[0]
    return frame, path


comments_path = get_source_csv(COMMENTS_PATH_OVERRIDE)
official_manifest, manifest_path = load_official_manifest(MANIFEST_PATH_OVERRIDE)
print('Verified source:', comments_path)
print('Official manifest:', manifest_path)
display(pd.DataFrame(split_profile(official_manifest)).T)

# %% [markdown]
# ## preprocessing ثابت و نمونه‌گیری group-safe
#
# train قدیمی دقیقاً حفظ می‌شود. validation/test قدیمی دقیقاً قفل می‌مانند. برای افزایش Macro-F1،
# داده‌های جدید فقط از گروه‌های متنی کاملاً جدید می‌آیند، گروه جدید متناقض کنار گذاشته می‌شود و
# انتخاب با quota کلاسی train قبلی به‌صورت اتمی انجام می‌شود. این سیاست عمداً با sampling سادهٔ v1
# یکسان نیست و اثر مشترک «دادهٔ بیشتر + پاک‌سازی گروهی» را می‌سنجد.

# %%
TEXT_COLUMNS = ['id', 'title', 'body', 'advantages', 'disadvantages', 'recommendation_status', 'product_id']
NULL_TOKENS = {'', 'nan', 'none', 'null', 'na', 'n/a'}
ARABIC_TO_PERSIAN = str.maketrans({'ي': 'ی', 'ى': 'ی', 'ك': 'ک'})
LABEL_BITS = {'recommended': 1, 'not_recommended': 2, 'no_idea': 4}
BIT_LABELS = {value: key for key, value in LABEL_BITS.items()}


def normalize_text_series(series: pd.Series) -> pd.Series:
    out = series.fillna('').astype(str)
    stripped_lower = out.str.strip().str.lower()
    out = out.mask(stripped_lower.isin(NULL_TOKENS), '')
    out = out.str.normalize('NFKC').str.translate(ARABIC_TO_PERSIAN)
    out = out.str.replace('\ufeff', '', regex=False)
    return out.str.replace(r'\s+', ' ', regex=True).str.strip()


def build_model_text(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ['title', 'body', 'advantages', 'disadvantages']:
        frame[column] = normalize_text_series(frame[column])
    full = pd.Series('', index=frame.index, dtype=object)
    for column, tag in [
        ('title', '[TITLE]'), ('body', '[BODY]'),
        ('advantages', '[ADVANTAGES]'), ('disadvantages', '[DISADVANTAGES]'),
    ]:
        full = full + np.where(frame[column].ne(''), tag + ' ' + frame[column] + ' ', '')
    frame['text_full'] = full.str.replace(r'\s+', ' ', regex=True).str.strip()
    frame['text_body'] = frame['body']
    split_text = frame['text_body'].where(frame['text_body'].ne(''), frame['text_full'])
    frame['text_group_id'] = split_text.map(lambda value: hashlib.sha1(value.encode('utf-8')).hexdigest())
    return frame


def stable_unit_hash(value: str, seed: int, namespace: str) -> float:
    raw = f'{namespace}|{seed}|{value}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') / 2**64


def canonical_digest(frame: pd.DataFrame) -> str:
    columns = ['id', 'product_id', 'text_group_id', 'recommendation_status', 'split']
    lines = frame[columns].astype(str).sort_values(columns).agg('\x1f'.join, axis=1)
    return hashlib.sha256('\n'.join(lines).encode('utf-8')).hexdigest()


frozen_ids = set(official_manifest['id'])
frozen_groups = set(official_manifest['text_group_id'])
frozen_train = official_manifest[official_manifest['split'].eq('train')].copy()
frozen_train_counts = frozen_train['recommendation_status'].value_counts().reindex(VALID_LABELS, fill_value=0)
frozen_train_ratios = frozen_train_counts / len(frozen_train)

# pass اول: audit کل داده و metadata گروه‌های candidate؛ متن کامل candidateها در RAM نگه داشته نمی‌شود.
seen_ids: set[str] = set()
candidate_meta: dict[str, list[int]] = {}
scan_rows = valid_label_rows = duplicate_ids_removed = empty_text_rows = 0
for chunk_number, chunk in enumerate(pd.read_csv(
    comments_path, usecols=TEXT_COLUMNS, dtype=str, chunksize=CHUNK_SIZE,
    keep_default_na=False, na_filter=False, encoding='utf-8-sig',
), start=1):
    scan_rows += len(chunk)
    chunk['recommendation_status'] = chunk['recommendation_status'].astype(str).str.strip()
    chunk = chunk[chunk['recommendation_status'].isin(VALID_LABELS)].copy()
    valid_label_rows += len(chunk)

    duplicate_mask = chunk['id'].isin(seen_ids) | chunk['id'].duplicated(keep='first')
    duplicate_ids_removed += int(duplicate_mask.sum())
    current_ids = chunk.loc[~duplicate_mask, 'id'].tolist()
    seen_ids.update(current_ids)
    chunk = chunk.loc[~duplicate_mask].copy()
    if chunk.empty:
        continue

    chunk = build_model_text(chunk)
    empty_text_rows += int(chunk['text_full'].eq('').sum())
    chunk = chunk[chunk['text_full'].ne('')].copy()
    candidates = chunk[~chunk['text_group_id'].isin(frozen_groups)].copy()
    if not candidates.empty:
        group_hash = candidates['text_group_id'].map(
            lambda value: stable_unit_hash(value, SEED, 'candidate-pool')
        )
        candidates = candidates[group_hash.lt(CANDIDATE_POOL_FRACTION)]
        for group_id, group in candidates.groupby('text_group_id', sort=False):
            bitmask = 0
            for label in group['recommendation_status'].unique():
                bitmask |= LABEL_BITS[label]
            if group_id in candidate_meta:
                candidate_meta[group_id][0] += len(group)
                candidate_meta[group_id][1] |= bitmask
            else:
                candidate_meta[group_id] = [int(len(group)), int(bitmask)]
    if chunk_number % 5 == 0:
        print(
            f'Pass 1 | chunks={chunk_number:,} scanned={scan_rows:,} '
            f'unique_valid_nonempty={len(seen_ids)-empty_text_rows:,} candidate_groups={len(candidate_meta):,}'
        )

valid_unique_nonempty_rows = len(seen_ids) - empty_text_rows
target_train_rows = int(round(TARGET_TRAIN_FRACTION * valid_unique_nonempty_rows))
if target_train_rows <= len(frozen_train):
    raise RuntimeError('Target 10% train is not larger than frozen 2% train.')

candidate_rows = pd.DataFrame([
    {'text_group_id': group_id, 'rows': values[0], 'label_bitmask': values[1]}
    for group_id, values in candidate_meta.items()
])
conflicting = candidate_rows[~candidate_rows['label_bitmask'].isin(BIT_LABELS)].copy()
clean_candidates = candidate_rows[candidate_rows['label_bitmask'].isin(BIT_LABELS)].copy()
clean_candidates['recommendation_status'] = clean_candidates['label_bitmask'].map(BIT_LABELS)
clean_candidates['order_hash'] = clean_candidates['text_group_id'].map(
    lambda value: stable_unit_hash(value, SEED, 'group-order')
)

# quotaها عمداً minority-enriched هستند تا دادهٔ واقعی بیشتری برای کلاس‌های ضعیف فراهم شود.
target_class_counts = {
    label: int(round(target_train_rows * TARGET_TRAIN_CLASS_RATIOS[label]))
    for label in VALID_LABELS
}
target_class_counts[VALID_LABELS[-1]] += target_train_rows - sum(target_class_counts.values())
additional_quota = {
    label: target_class_counts[label] - int(frozen_train_counts[label])
    for label in VALID_LABELS
}


def choose_groups_nearest(frame: pd.DataFrame, quota_rows: int) -> tuple[list[str], int]:
    frame = frame.sort_values(['order_hash', 'text_group_id']).reset_index(drop=True)
    if int(frame['rows'].sum()) < quota_rows:
        raise RuntimeError(
            f'Candidate pool is too small: available={int(frame["rows"].sum()):,}, needed={quota_rows:,}. '
            'Increase CANDIDATE_POOL_FRACTION and rerun before training.'
        )
    chosen, total = [], 0
    for row in frame.itertuples(index=False):
        before = abs(quota_rows - total)
        after = abs(quota_rows - (total + int(row.rows)))
        if total < quota_rows and after <= before:
            chosen.append(row.text_group_id)
            total += int(row.rows)
        elif total >= quota_rows:
            break
        # یک گروه بزرگ که overshoot بدی می‌دهد کنار گذاشته می‌شود؛ ممکن است گروه کوچک‌تری
        # بعدتر quota را دقیق‌تر پر کند.
    return chosen, total


selected_new_groups: set[str] = set()
selected_rows_by_label = {}
for label in VALID_LABELS:
    chosen, rows = choose_groups_nearest(
        clean_candidates[clean_candidates['recommendation_status'].eq(label)],
        additional_quota[label],
    )
    selected_new_groups.update(chosen)
    selected_rows_by_label[label] = rows

print(json.dumps({
    'valid_unique_nonempty_rows': valid_unique_nonempty_rows,
    'target_train_rows': target_train_rows,
    'frozen_train_rows': len(frozen_train),
    'candidate_groups': len(candidate_rows),
    'conflicting_candidate_groups': len(conflicting),
    'additional_quota': additional_quota,
    'selected_rows_by_label': selected_rows_by_label,
}, ensure_ascii=False, indent=2))

# %% [markdown]
# ## pass دوم: بازیابی دقیق رکوردهای انتخاب‌شده و اثبات قفل‌بودن splitها

# %%
official_by_id = official_manifest.set_index('id')
seen_ids = set()
selected_chunks = []
for chunk_number, chunk in enumerate(pd.read_csv(
    comments_path, usecols=TEXT_COLUMNS, dtype=str, chunksize=CHUNK_SIZE,
    keep_default_na=False, na_filter=False, encoding='utf-8-sig',
), start=1):
    chunk['recommendation_status'] = chunk['recommendation_status'].astype(str).str.strip()
    chunk = chunk[chunk['recommendation_status'].isin(VALID_LABELS)].copy()
    duplicate_mask = chunk['id'].isin(seen_ids) | chunk['id'].duplicated(keep='first')
    current_ids = chunk.loc[~duplicate_mask, 'id'].tolist()
    seen_ids.update(current_ids)
    chunk = chunk.loc[~duplicate_mask].copy()
    if chunk.empty:
        continue
    chunk = build_model_text(chunk)
    chunk = chunk[chunk['text_full'].ne('')].copy()
    keep = chunk['id'].isin(frozen_ids) | chunk['text_group_id'].isin(selected_new_groups)
    chosen = chunk[keep].copy()
    if not chosen.empty:
        selected_chunks.append(chosen)
    if chunk_number % 5 == 0:
        print(f'Pass 2 | chunks={chunk_number:,} recovered_rows={sum(map(len, selected_chunks)):,}')

data = pd.concat(selected_chunks, ignore_index=True)
if data['id'].duplicated().any():
    raise RuntimeError('Duplicate IDs survived pass 2.')

locked = data[data['id'].isin(frozen_ids)].copy()
if set(locked['id']) != frozen_ids:
    missing = sorted(frozen_ids - set(locked['id']))[:10]
    raise RuntimeError(f'Frozen IDs were not recovered: {missing}')

source_check = locked[['id', 'product_id', 'text_group_id', 'recommendation_status']].merge(
    official_manifest[['id', 'product_id', 'text_group_id', 'recommendation_status']],
    on='id', suffixes=('_source', '_official'), validate='one_to_one',
)
for column in ['product_id', 'text_group_id', 'recommendation_status']:
    if not source_check[f'{column}_source'].astype(str).eq(source_check[f'{column}_official'].astype(str)).all():
        raise RuntimeError(f'Frozen source/manifest mismatch in {column}.')

data = data.merge(official_manifest[['id', 'split']], on='id', how='left', validate='one_to_one')
data['split'] = data['split'].fillna('train')
data['label_id'] = data['recommendation_status'].map(LABEL2ID).astype(int)

new_rows = data[~data['id'].isin(frozen_ids)]
if not set(new_rows['text_group_id']).issubset(selected_new_groups):
    raise RuntimeError('Unexpected new group was recovered.')
if set(new_rows['text_group_id']) & frozen_groups:
    raise RuntimeError('A frozen group leaked into newly added train rows.')
if new_rows.groupby('text_group_id')['recommendation_status'].nunique().gt(1).any():
    raise RuntimeError('A conflicting new group survived selection.')
observed_added = (
    new_rows['recommendation_status'].value_counts().reindex(VALID_LABELS, fill_value=0).astype(int).to_dict()
)
if observed_added != selected_rows_by_label:
    raise RuntimeError(
        f'Pass-1/pass-2 selected-row mismatch: expected={selected_rows_by_label}, observed={observed_added}'
    )

train_df = data[data['split'].eq('train')].copy().reset_index(drop=True)
val_df = data[data['split'].eq('validation')].copy().reset_index(drop=True)
test_df = data[data['split'].eq('test')].copy().reset_index(drop=True)
expected_train_rows = len(frozen_train) + sum(selected_rows_by_label.values())
if len(train_df) != expected_train_rows:
    raise RuntimeError(f'Unexpected train size: observed={len(train_df)}, expected={expected_train_rows}')
row_delta = abs(len(train_df) - target_train_rows)
largest_clean_group = int(clean_candidates['rows'].max())
if row_delta > max(largest_clean_group, int(0.002 * target_train_rows)):
    raise RuntimeError(f'10% target missed beyond tolerance: delta={row_delta}')

for left_name, left, right_name, right in [
    ('train', train_df, 'validation', val_df),
    ('train', train_df, 'test', test_df),
    ('validation', val_df, 'test', test_df),
]:
    if set(left['id']) & set(right['id']):
        raise RuntimeError(f'ID leakage: {left_name}/{right_name}')
    if set(left['text_group_id']) & set(right['text_group_id']):
        raise RuntimeError(f'Group leakage: {left_name}/{right_name}')

if not set(frozen_train['id']).issubset(set(train_df['id'])):
    raise RuntimeError('Old 2% train is not a complete subset of 10% train.')
if set(val_df['id']) != set(official_manifest.loc[official_manifest['split'].eq('validation'), 'id']):
    raise RuntimeError('Validation IDs changed.')
if set(test_df['id']) != set(official_manifest.loc[official_manifest['split'].eq('test'), 'id']):
    raise RuntimeError('Test IDs changed.')

final_manifest = data[['id', 'product_id', 'text_group_id', 'recommendation_status', 'split']].copy()
official_val_digest = canonical_digest(official_manifest[official_manifest['split'].eq('validation')])
official_test_digest = canonical_digest(official_manifest[official_manifest['split'].eq('test')])
if canonical_digest(final_manifest[final_manifest['split'].eq('validation')]) != official_val_digest:
    raise RuntimeError('Validation digest changed.')
if canonical_digest(final_manifest[final_manifest['split'].eq('test')]) != official_test_digest:
    raise RuntimeError('Test digest changed.')

expanded_profile = split_profile(final_manifest)
display(pd.DataFrame(expanded_profile).T)

manifest_output_path = OUTPUT_DIR / 'recommendation_10pct_split_manifest.csv'
audit_path = OUTPUT_DIR / 'recommendation_10pct_data_audit.json'
conflict_path = OUTPUT_DIR / 'recommendation_10pct_conflicting_groups.csv'
final_manifest.to_csv(manifest_output_path, index=False)
conflicting.to_csv(conflict_path, index=False)

sampling_audit = {
    'algorithm': 'two_pass_group_stratified_sha256_v1',
    'seed': SEED,
    'target_train_fraction': TARGET_TRAIN_FRACTION,
    'candidate_pool_fraction': CANDIDATE_POOL_FRACTION,
    'source_rows_scanned': scan_rows,
    'valid_label_rows': valid_label_rows,
    'duplicate_id_rows_removed': duplicate_ids_removed,
    'empty_text_rows_removed': empty_text_rows,
    'valid_unique_nonempty_rows': valid_unique_nonempty_rows,
    'target_train_rows': target_train_rows,
    'actual_train_rows': len(train_df),
    'actual_train_fraction': float(len(train_df) / valid_unique_nonempty_rows),
    'frozen_train_rows': len(frozen_train),
    'added_train_rows': len(train_df) - len(frozen_train),
    'candidate_groups': len(candidate_rows),
    'candidate_rows': int(candidate_rows['rows'].sum()),
    'conflicting_candidate_groups': len(conflicting),
    'conflicting_candidate_rows': int(conflicting['rows'].sum()),
    'selected_new_groups': len(selected_new_groups),
    'target_train_class_ratios': TARGET_TRAIN_CLASS_RATIOS,
    'selected_rows_by_label': selected_rows_by_label,
    'expanded_profile': expanded_profile,
    'validation_digest': official_val_digest,
    'test_digest': official_test_digest,
}
audit_path.write_text(json.dumps(sampling_audit, ensure_ascii=False, indent=2), encoding='utf-8')

# آزادکردن اشیای بزرگ sampling پیش از ساخت مدل؛ فقط ستون‌های موردنیاز train/eval می‌مانند.
model_columns = [
    'id', 'product_id', 'text_group_id', 'recommendation_status', 'text_full', 'label_id',
    'title', 'body', 'advantages', 'disadvantages',
]
train_df = train_df[model_columns].copy()
val_df = val_df[model_columns].copy()
test_df = test_df[model_columns].copy()
del candidate_meta, candidate_rows, clean_candidates, selected_chunks, data
del locked, source_check, new_rows, seen_ids, conflicting
gc.collect()

# %% [markdown]
# ## Dataset، معیارها، fingerprint و resume

# %%
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True).astype(int)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = self.tokenizer(
            self.texts.iloc[index], truncation=True, max_length=self.max_length, padding=False,
        )
        item['labels'] = int(self.labels.iloc[index])
        return item


def calculate_metrics(y_true_ids, y_pred_ids) -> dict:
    y_true_ids = np.asarray(y_true_ids, dtype=int)
    y_pred_ids = np.asarray(y_pred_ids, dtype=int)
    result = {
        'macro_f1': float(f1_score(y_true_ids, y_pred_ids, labels=list(ID2LABEL), average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true_ids, y_pred_ids, labels=list(ID2LABEL), average='weighted', zero_division=0)),
        'accuracy': float(accuracy_score(y_true_ids, y_pred_ids)),
    }
    report = classification_report(
        y_true_ids, y_pred_ids, labels=list(ID2LABEL), output_dict=True, zero_division=0,
    )
    for label_id, label_name in ID2LABEL.items():
        row = report[str(label_id)]
        result[f'precision_{label_name}'] = float(row['precision'])
        result[f'recall_{label_name}'] = float(row['recall'])
        result[f'f1_{label_name}'] = float(row['f1-score'])
        result[f'support_{label_name}'] = int(row['support'])
    return result


def trainer_metrics(eval_prediction):
    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    return calculate_metrics(labels, np.argmax(logits, axis=-1))


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = nn.CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def class_weights_for(frame: pd.DataFrame) -> list[float]:
    weights = compute_class_weight(
        class_weight='balanced', classes=np.arange(len(VALID_LABELS)), y=frame['label_id'].to_numpy(),
    )
    return [float(value) for value in weights]


def make_training_arguments(output_dir: Path, epochs: float, do_eval: bool) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = 'eval_strategy' if 'eval_strategy' in signature else 'evaluation_strategy'
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(epochs),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type='linear',
        fp16=True,
        bf16=False,
        logging_steps=100,
        report_to='none',
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=2,
        save_strategy='steps',
        save_steps=EVAL_SAVE_STEPS,
        save_total_limit=2,
        save_only_model=False,
        remove_unused_columns=True,
        ignore_data_skip=False,
    )
    if do_eval:
        kwargs.update({
            eval_key: 'steps',
            'eval_steps': EVAL_SAVE_STEPS,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'macro_f1',
            'greater_is_better': True,
        })
    else:
        kwargs.update({eval_key: 'no', 'load_best_model_at_end': False})
    return TrainingArguments(**kwargs)


def make_trainer(model, tokenizer, train_dataset, eval_dataset, epochs, class_weights):
    kwargs = dict(
        model=model,
        args=make_training_arguments(TRAIN_RUN_DIR, epochs, eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors='pt',
        ),
        compute_metrics=trainer_metrics if eval_dataset is not None else None,
        class_weights=class_weights,
    )
    if 'processing_class' in inspect.signature(Trainer.__init__).parameters:
        kwargs['processing_class'] = tokenizer
    else:
        kwargs['tokenizer'] = tokenizer
    return WeightedTrainer(**kwargs)


manifest_digest = canonical_digest(final_manifest)


def resolve_passed_validation_summary() -> tuple[dict, Path] | tuple[None, None]:
    if RUN_PHASE != 'final':
        return None, None
    if VALIDATION_SUMMARY_OVERRIDE:
        candidates = [Path(VALIDATION_SUMMARY_OVERRIDE)]
    else:
        candidates = []
        for root in [Path('/kaggle/working'), Path('/kaggle/input'), Path.cwd()]:
            if root.exists():
                candidates.extend(root.rglob('recommendation_10pct_validation_summary.json'))
    compatible = []
    for candidate in candidates:
        try:
            value = json.loads(candidate.read_text(encoding='utf-8'))
            required = (
                value.get('validation_gate_passed') is True
                and value.get('decision') == 'PASS_TO_FINAL'
                and value.get('source_sha256') == HF_EXPECTED_SHA256
                and value.get('expanded_manifest_digest') == manifest_digest
                and value.get('model') == MODEL_NAME
                and value.get('model_revision') == MODEL_REVISION
                and int(value.get('max_length', -1)) == MAX_LENGTH
                and value.get('preprocessing') == PREPROCESSING_VERSION
            )
            epochs = float(value.get('selected_epochs_for_final', 0))
            if required and 0 < epochs <= EPOCHS:
                compatible.append((value, candidate))
        except Exception:
            pass
    epoch_values = {float(value['selected_epochs_for_final']) for value, _ in compatible}
    if not compatible:
        raise FileNotFoundError('Compatible PASS validation summary is required for RUN_PHASE=final.')
    if len(epoch_values) != 1:
        raise RuntimeError(
            'Multiple compatible validation summaries disagree on selected epoch. '
            'Set DIGIKALA_VALIDATION_SUMMARY_PATH explicitly.'
        )
    # چند کپی یکسان (مثلاً working و input) بی‌ضرر است.
    return compatible[0]


resolved_validation_summary, resolved_validation_summary_path = resolve_passed_validation_summary()
final_selected_epochs_hint = (
    float(resolved_validation_summary['selected_epochs_for_final'])
    if resolved_validation_summary is not None else None
)

run_fingerprint = {
    'phase': RUN_PHASE,
    'source_sha256': HF_EXPECTED_SHA256,
    'official_manifest_path_name': manifest_path.name,
    'expanded_manifest_digest': manifest_digest,
    'train_rows': len(train_df),
    'validation_rows': len(val_df),
    'test_rows': len(test_df),
    'seed': SEED,
    'labels': VALID_LABELS,
    'model': MODEL_NAME,
    'model_revision': MODEL_REVISION,
    'preprocessing': PREPROCESSING_VERSION,
    'max_length': MAX_LENGTH,
    'epochs': EPOCHS,
    'selected_epochs_for_final': final_selected_epochs_hint,
    'train_batch_size': TRAIN_BATCH_SIZE,
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'warmup_ratio': WARMUP_RATIO,
}
fingerprint_path = TRAIN_RUN_DIR / 'run_fingerprint.json'
local_checkpoints = list(TRAIN_RUN_DIR.glob('checkpoint-*'))
if fingerprint_path.exists():
    existing_fingerprint = json.loads(fingerprint_path.read_text(encoding='utf-8'))
    if local_checkpoints and existing_fingerprint != run_fingerprint:
        raise RuntimeError(
            'Local checkpoints belong to a different run fingerprint. '
            'Use a fresh Kaggle session/output directory instead of mixing runs.'
        )
fingerprint_path.write_text(json.dumps(run_fingerprint, ensure_ascii=False, indent=2), encoding='utf-8')


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit('-', 1)[1])
    except Exception:
        return -1


def checkpoint_fingerprint(path: Path) -> dict | None:
    for parent in [path.parent, path.parent.parent, path.parent.parent.parent]:
        candidate = parent / 'run_fingerprint.json'
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding='utf-8'))
            except Exception:
                return None
    return None


def find_resume_checkpoint() -> Path | None:
    if RESUME_CHECKPOINT_OVERRIDE:
        path = Path(RESUME_CHECKPOINT_OVERRIDE)
        if not path.is_dir():
            raise FileNotFoundError(path)
        if checkpoint_fingerprint(path) != run_fingerprint:
            raise RuntimeError('Explicit checkpoint fingerprint does not match this run.')
        return path
    candidates = []
    for root in [TRAIN_RUN_DIR, Path('/kaggle/input')]:
        if root.exists():
            candidates.extend(path for path in root.rglob('checkpoint-*') if path.is_dir())
    compatible = [path for path in candidates if checkpoint_fingerprint(path) == run_fingerprint]
    return max(compatible, key=checkpoint_step) if compatible else None


resume_checkpoint = find_resume_checkpoint()
if resume_checkpoint is not None and Path('/kaggle/input') in resume_checkpoint.parents:
    # Saved Version read-only است و trainer_state مسیر مطلق session قبلی را نگه می‌دارد.
    # دو checkpoint سازگار را به همان ساختار local می‌آوریم و best path را patch می‌کنیم.
    source_siblings = [
        path for path in resume_checkpoint.parent.glob('checkpoint-*')
        if path.is_dir() and checkpoint_fingerprint(path) == run_fingerprint
    ]
    for source_checkpoint in source_siblings:
        local_checkpoint = TRAIN_RUN_DIR / source_checkpoint.name
        if not local_checkpoint.exists():
            print('Copying resumable checkpoint to working:', source_checkpoint)
            shutil.copytree(source_checkpoint, local_checkpoint)
        state_path = local_checkpoint / 'trainer_state.json'
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            old_best = state.get('best_model_checkpoint')
            if old_best:
                local_best = TRAIN_RUN_DIR / Path(old_best).name
                if local_best.is_dir():
                    state['best_model_checkpoint'] = str(local_best)
                    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
                else:
                    raise RuntimeError(
                        f'Saved Version is missing its recorded best checkpoint: {Path(old_best).name}'
                    )
    resume_checkpoint = TRAIN_RUN_DIR / resume_checkpoint.name
print('Resume checkpoint:', resume_checkpoint or 'none — fresh training')

# %% [markdown]
# ## فاز validation
#
# این بخش test را predict نمی‌کند. هشدار initializeشدن classifier طبیعی است؛ head سه‌کلاسه باید
# روی task ما آموزش ببیند. همچنین `fix_mistral_regex` عمداً به tokenizer پاس داده نمی‌شود.

# %%
validation_summary = None
if RUN_PHASE == 'validation':
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION,
        num_labels=len(VALID_LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    resolved_revision = getattr(model.config, '_commit_hash', None)
    if resolved_revision and resolved_revision != MODEL_REVISION:
        raise RuntimeError(f'Model resolved to unexpected revision: {resolved_revision}')

    train_dataset = TextClassificationDataset(train_df['text_full'], train_df['label_id'], tokenizer, MAX_LENGTH)
    validation_dataset = TextClassificationDataset(val_df['text_full'], val_df['label_id'], tokenizer, MAX_LENGTH)
    train_weights = class_weights_for(train_df)
    trainer = make_trainer(
        model, tokenizer, train_dataset, validation_dataset, EPOCHS, train_weights,
    )

    started = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    fit_seconds = float(time.perf_counter() - started)
    prediction_output = trainer.predict(validation_dataset)
    logits = prediction_output.predictions[0] if isinstance(prediction_output.predictions, tuple) else prediction_output.predictions
    prediction_ids = np.argmax(logits, axis=-1)
    scores = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    validation_metrics = calculate_metrics(val_df['label_id'], prediction_ids)

    eval_logs = [row for row in trainer.state.log_history if 'eval_macro_f1' in row]
    best_log = max(eval_logs, key=lambda row: row['eval_macro_f1']) if eval_logs else {'epoch': EPOCHS, 'step': trainer.state.global_step}
    selected_epochs = float(best_log.get('epoch') or EPOCHS)
    selected_step = int(best_log.get('step') or trainer.state.global_step)

    gates = {
        'macro_f1_gain_at_least_0.005': validation_metrics['macro_f1'] >= V1_VALIDATION['macro_f1'] + 0.005,
        'validation_rows_locked': len(val_df) == EXPECTED_2PCT_PROFILE['validation']['rows'],
        'test_rows_untouched': len(test_df) == EXPECTED_2PCT_PROFILE['test']['rows'],
    }
    advisory_class_checks = {
        'recommended_f1_regression_at_most_0.02': validation_metrics['f1_recommended'] >= V1_VALIDATION['f1_recommended'] - 0.02,
        'not_recommended_f1_regression_at_most_0.02': validation_metrics['f1_not_recommended'] >= V1_VALIDATION['f1_not_recommended'] - 0.02,
        'no_idea_f1_regression_at_most_0.02': validation_metrics['f1_no_idea'] >= V1_VALIDATION['f1_no_idea'] - 0.02,
    }
    validation_gate_passed = all(gates.values())

    best_dir = OUTPUT_DIR / 'best_transformer_encoder_10pct_validation'
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (best_dir / 'inference_config.json').write_text(json.dumps({
        'model_version': MODEL_RC_VERSION,
        'status': 'EXPERIMENT',
        'labels': VALID_LABELS,
        'max_length': MAX_LENGTH,
        'normalization_version': PREPROCESSING_VERSION,
        'text_column': 'text_full',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    validation_predictions = val_df[['id', 'product_id', 'text_group_id', 'recommendation_status']].copy()
    validation_predictions['prediction_10pct'] = [ID2LABEL[int(value)] for value in prediction_ids]
    for label_id, label in ID2LABEL.items():
        validation_predictions[f'score_{label}_10pct'] = scores[:, label_id]
    validation_predictions.to_csv(OUTPUT_DIR / 'recommendation_10pct_validation_predictions.csv', index=False)
    pd.DataFrame(trainer.state.log_history).to_csv(OUTPUT_DIR / 'recommendation_10pct_training_history.csv', index=False)

    validation_summary = {
        'task': 'digikala_recommendation_status_10pct_validation',
        'decision': 'PASS_TO_FINAL' if validation_gate_passed else 'FAIL_KEEP_V1',
        'validation_gate_passed': validation_gate_passed,
        'model_version': MODEL_RC_VERSION,
        'source_sha256': HF_EXPECTED_SHA256,
        'expanded_manifest_digest': manifest_digest,
        'model': MODEL_NAME,
        'model_revision': MODEL_REVISION,
        'max_length': MAX_LENGTH,
        'preprocessing': PREPROCESSING_VERSION,
        'selected_epochs_for_final': selected_epochs,
        'selected_step': selected_step,
        'fit_seconds_this_session': fit_seconds,
        'trainer_train_metrics': train_result.metrics,
        'class_weights': dict(zip(VALID_LABELS, train_weights)),
        'sampling_audit': sampling_audit,
        'v1_validation_reference': V1_VALIDATION,
        'validation_metrics': validation_metrics,
        'absolute_macro_f1_gain': validation_metrics['macro_f1'] - V1_VALIDATION['macro_f1'],
        'gates': gates,
        'advisory_class_checks': advisory_class_checks,
        'runtime': runtime_info,
        'best_validation_model_dir': str(best_dir),
    }
    validation_summary_path = OUTPUT_DIR / 'recommendation_10pct_validation_summary.json'
    validation_summary_path.write_text(json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n' + '#' * 24 + ' COPY THIS VALIDATION SUMMARY ' + '#' * 24)
    print(json.dumps(validation_summary, ensure_ascii=False, indent=2))
    print('\nNext action:', 'Save Version, attach it as Input, set RUN_PHASE=final.' if validation_gate_passed else 'Keep v1; do not run final/test.')
else:
    print('RUN_PHASE=final: validation training cell skipped.')

# %% [markdown]
# ## فاز final و test (فقط پس از PASS validation)
#
# این فاز summary اجرای validation را اجباری می‌کند، از checkpoint پایه روی `train + validation`
# برای epoch انتخاب‌شده آموزش می‌دهد و test را یک بار ارزیابی می‌کند. برای promotion، فایل
# prediction رسمی v1 نیز اجباری است تا paired bootstrap در سطح `text_group_id` انجام شود.

# %%
def load_validation_summary() -> tuple[dict, str]:
    if resolved_validation_summary is None or resolved_validation_summary_path is None:
        raise FileNotFoundError('Compatible PASS validation summary was not resolved.')
    return resolved_validation_summary, str(resolved_validation_summary_path)


def load_v1_test_predictions() -> tuple[pd.DataFrame, str, str]:
    found = []
    if V1_PREDICTIONS_OVERRIDE:
        direct = Path(V1_PREDICTIONS_OVERRIDE)
        roots = [direct]
    else:
        roots = [Path('/kaggle/input'), Path('/kaggle/working'), Path.cwd()]
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() and root.name == 'recommendation_test_predictions.csv' else list(root.rglob('recommendation_test_predictions.csv'))
        for path in paths:
            try:
                raw = path.read_bytes()
                found.append((pd.read_csv(io.BytesIO(raw), dtype={'id': str}, keep_default_na=False), str(path), hashlib.sha256(raw).hexdigest()))
            except Exception:
                pass
        zip_paths = [root] if root.is_file() and root.suffix.lower() == '.zip' else list(root.rglob('*.zip'))
        for path in zip_paths:
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in [name for name in archive.namelist() if name.endswith('recommendation_test_predictions.csv')]:
                        raw = archive.read(member)
                        found.append((pd.read_csv(io.BytesIO(raw), dtype={'id': str}, keep_default_na=False), f'{path}::{member}', hashlib.sha256(raw).hexdigest()))
            except (zipfile.BadZipFile, OSError):
                pass
    if not found:
        raise FileNotFoundError(
            'recommendation_test_predictions.csv نسخه v1 پیدا نشد. Dataset عمومی v1 را Add Input کنید.'
        )
    distinct_hashes = {item[2] for item in found}
    if len(distinct_hashes) > 1 and not V1_PREDICTIONS_OVERRIDE:
        raise RuntimeError(
            'چند prediction artifact متفاوت برای v1 پیدا شد. '
            'مسیر درست را در DIGIKALA_V1_PREDICTIONS_PATH مشخص کنید.'
        )
    return found[0]


def confusion_by_group(frame: pd.DataFrame, prediction_ids: np.ndarray) -> np.ndarray:
    group_codes, unique_groups = pd.factorize(frame['text_group_id'], sort=True)
    matrices = np.zeros((len(unique_groups), len(VALID_LABELS), len(VALID_LABELS)), dtype=np.int32)
    np.add.at(matrices, (group_codes, frame['label_id'].to_numpy(), prediction_ids), 1)
    return matrices


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    true_positive = np.diag(matrix).astype(float)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    class_f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator != 0)
    return float(class_f1.mean())


final_summary = None
if RUN_PHASE == 'final':
    prior_validation, prior_validation_source = load_validation_summary()
    if not prior_validation.get('validation_gate_passed'):
        raise RuntimeError('Validation gate did not pass; test must remain unopened and v1 must be kept.')
    selected_epochs = float(prior_validation['selected_epochs_for_final'])

    v1_predictions, v1_predictions_source, v1_predictions_sha256 = load_v1_test_predictions()
    v1_prediction_column = next((name for name in [
        'transformer_prediction', 'prediction_2pct', 'prediction'
    ] if name in v1_predictions.columns), None)
    if v1_prediction_column is None:
        raise RuntimeError(f'No v1 prediction column found: {list(v1_predictions.columns)}')
    required_v1 = {'id', 'text_group_id', 'recommendation_status', v1_prediction_column}
    if not required_v1.issubset(v1_predictions.columns):
        raise RuntimeError(f'Incomplete v1 prediction artifact: {required_v1 - set(v1_predictions.columns)}')

    reference = test_df[['id', 'text_group_id', 'recommendation_status']].merge(
        v1_predictions[list(required_v1)], on='id', suffixes=('_locked', '_v1'), validate='one_to_one',
    )
    if len(reference) != len(test_df):
        raise RuntimeError('v1 predictions do not cover the exact locked test.')
    if reference['id'].tolist() != test_df['id'].tolist():
        raise RuntimeError('v1 alignment changed locked test row order.')
    for column in ['text_group_id', 'recommendation_status']:
        if not reference[f'{column}_locked'].eq(reference[f'{column}_v1']).all():
            raise RuntimeError(f'v1 prediction artifact mismatch in {column}.')
    v1_prediction_ids = reference[v1_prediction_column].map(LABEL2ID)
    if v1_prediction_ids.isna().any():
        raise RuntimeError('Unknown label in v1 predictions.')
    v1_prediction_ids = v1_prediction_ids.to_numpy(dtype=int)
    reproduced_v1_metrics = calculate_metrics(test_df['label_id'], v1_prediction_ids)
    v1_reproduction_delta = abs(reproduced_v1_metrics['macro_f1'] - V1_TEST['macro_f1'])
    if v1_reproduction_delta > 0.001:
        raise RuntimeError(f'v1 Macro-F1 reproduction delta too large: {v1_reproduction_delta}')

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION,
        num_labels=len(VALID_LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_dataset = TextClassificationDataset(
        train_val_df['text_full'], train_val_df['label_id'], tokenizer, MAX_LENGTH,
    )
    test_dataset = TextClassificationDataset(test_df['text_full'], test_df['label_id'], tokenizer, MAX_LENGTH)
    final_weights = class_weights_for(train_val_df)
    trainer = make_trainer(model, tokenizer, train_val_dataset, None, selected_epochs, final_weights)

    started = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    final_fit_seconds = float(time.perf_counter() - started)
    trainer_train_metrics = dict(train_result.metrics)

    predict_started = time.perf_counter()
    output = trainer.predict(test_dataset)
    test_predict_seconds = float(time.perf_counter() - predict_started)
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    prediction_ids = np.argmax(logits, axis=-1)
    scores = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    metrics = calculate_metrics(test_df['label_id'], prediction_ids)

    # paired group bootstrap: در هر iteration همان گروه‌ها با همان multiplicity برای هر دو مدل.
    new_group_cm = confusion_by_group(test_df, prediction_ids)
    v1_group_cm = confusion_by_group(test_df, v1_prediction_ids)
    group_count = new_group_cm.shape[0]
    rng = np.random.default_rng(SEED)
    new_bootstrap = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    v1_bootstrap = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    probabilities = np.full(group_count, 1.0 / group_count)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        counts = rng.multinomial(group_count, probabilities)
        new_bootstrap[iteration] = macro_f1_from_confusion(np.tensordot(counts, new_group_cm, axes=(0, 0)))
        v1_bootstrap[iteration] = macro_f1_from_confusion(np.tensordot(counts, v1_group_cm, axes=(0, 0)))
    paired_delta = new_bootstrap - v1_bootstrap
    bootstrap = {
        'iterations': BOOTSTRAP_ITERATIONS,
        'seed': SEED,
        'resampling_unit': 'text_group_id',
        'new_macro_f1_ci95': [float(value) for value in np.quantile(new_bootstrap, [0.025, 0.975])],
        'v1_macro_f1_ci95': [float(value) for value in np.quantile(v1_bootstrap, [0.025, 0.975])],
        'paired_macro_f1_delta_ci95': [float(value) for value in np.quantile(paired_delta, [0.025, 0.975])],
        'paired_probability_new_better': float(np.mean(paired_delta > 0)),
    }

    # مدل inference تمیز ذخیره و reload می‌شود تا optimizer/state آموزشی وارد peak memory نشود.
    staging_model_dir = OUTPUT_DIR / 'best_transformer_encoder_10pct_staging'
    if staging_model_dir.exists():
        raise RuntimeError('Staging model directory already exists. Use a fresh session for final evaluation.')
    staging_model_dir.mkdir(parents=True, exist_ok=False)
    trainer.save_model(str(staging_model_dir))
    tokenizer.save_pretrained(str(staging_model_dir))
    trainer.optimizer = None
    trainer.lr_scheduler = None
    del trainer, model, train_result, train_val_dataset, test_dataset, output, logits
    gc.collect()
    torch.cuda.empty_cache()

    # latency با پروتکل نسخه v1؛ شامل tokenize + transfer + forward.
    device = torch.device('cuda')
    model = AutoModelForSequenceClassification.from_pretrained(
        staging_model_dir, local_files_only=True,
    ).to(device).eval()

    def timed_batch(texts) -> float:
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        encoded = tokenizer(
            list(texts), truncation=True, max_length=MAX_LENGTH,
            padding=True, pad_to_multiple_of=8, return_tensors='pt',
        ).to(device)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
            _ = model(**encoded).logits
        torch.cuda.synchronize()
        return (time.perf_counter() - started_at) * 1000

    latency_texts = test_df.sample(min(1024, len(test_df)), random_state=SEED)['text_full'].tolist()
    for _ in range(10):
        timed_batch(latency_texts[:1])
    torch.cuda.reset_peak_memory_stats()
    single_latencies = np.array([timed_batch([text]) for text in latency_texts[:200]])
    batch_latencies, batch_rows = [], 0
    batch_started = time.perf_counter()
    for start in range(0, len(latency_texts), 32):
        texts = latency_texts[start:start + 32]
        batch_latencies.append(timed_batch(texts))
        batch_rows += len(texts)
    batch_seconds = time.perf_counter() - batch_started
    latency = {
        'gpu': GPU_NAME,
        'single_samples': len(single_latencies),
        'single_request_latency_ms': {
            'mean': float(single_latencies.mean()),
            'p50': float(np.quantile(single_latencies, 0.50)),
            'p95': float(np.quantile(single_latencies, 0.95)),
            'p99': float(np.quantile(single_latencies, 0.99)),
        },
        'batch_size': 32,
        'batch_rows': batch_rows,
        'batch_throughput_rows_per_second': float(batch_rows / batch_seconds),
        'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated()),
    }

    same_runtime_as_v1 = (
        'T4' in GPU_NAME
        and torch.__version__ == '2.10.0+cu128'
        and transformers.__version__ == '4.57.6'
        and MAX_LENGTH == 128
    )
    gates = {
        'v1_metric_reproduction_within_0.001': v1_reproduction_delta <= 0.001,
        'test_macro_f1_gain_at_least_0.005': metrics['macro_f1'] >= reproduced_v1_metrics['macro_f1'] + 0.005,
        'paired_delta_ci95_lower_above_zero': bootstrap['paired_macro_f1_delta_ci95'][0] > 0,
        'single_request_p95_under_250ms': latency['single_request_latency_ms']['p95'] <= 250,
    }
    advisory_checks = {
        'recommended_f1_regression_at_most_0.02': metrics['f1_recommended'] >= reproduced_v1_metrics['f1_recommended'] - 0.02,
        'not_recommended_f1_regression_at_most_0.02': metrics['f1_not_recommended'] >= reproduced_v1_metrics['f1_not_recommended'] - 0.02,
        'no_idea_f1_non_regression': metrics['f1_no_idea'] >= reproduced_v1_metrics['f1_no_idea'],
        'runtime_exactly_comparable_to_v1': same_runtime_as_v1,
        'relative_single_p95_within_1.25x_v1': latency['single_request_latency_ms']['p95'] <= 1.25 * V1_TEST['single_p95_ms_t4'],
        'relative_batch_throughput_at_least_0.80x_v1': latency['batch_throughput_rows_per_second'] >= 0.80 * V1_TEST['batch_throughput_t4'],
        'relative_peak_memory_within_1.25x_v1': latency['peak_gpu_memory_bytes'] <= 1.25 * V1_TEST['peak_gpu_memory_bytes_t4'],
    }
    release_passed = all(gates.values())

    release_dir = OUTPUT_DIR / (
        'best_transformer_encoder_10pct' if release_passed else 'best_transformer_encoder_10pct_rc_failed'
    )
    if release_dir.exists():
        raise RuntimeError(f'Release target already exists; use a fresh session: {release_dir}')
    staging_model_dir.rename(release_dir)
    (release_dir / 'inference_config.json').write_text(json.dumps({
        'model_version': MODEL_RELEASE_VERSION if release_passed else MODEL_RC_VERSION,
        'status': 'RELEASED' if release_passed else 'EXPERIMENT',
        'labels': VALID_LABELS,
        'max_length': MAX_LENGTH,
        'normalization_version': PREPROCESSING_VERSION,
        'text_column': 'text_full',
        'schema_version': '1.0.0',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    predictions = test_df[['id', 'product_id', 'text_group_id', 'recommendation_status']].copy()
    predictions['prediction_2pct'] = [ID2LABEL[int(value)] for value in v1_prediction_ids]
    predictions['prediction_10pct'] = [ID2LABEL[int(value)] for value in prediction_ids]
    for label_id, label in ID2LABEL.items():
        predictions[f'score_{label}_10pct'] = scores[:, label_id]
    predictions.to_csv(OUTPUT_DIR / 'recommendation_10pct_test_predictions.csv', index=False)

    per_class = pd.DataFrame(classification_report(
        test_df['label_id'], prediction_ids, labels=list(ID2LABEL), target_names=VALID_LABELS,
        output_dict=True, zero_division=0,
    )).T.loc[VALID_LABELS].reset_index(names='label')
    per_class.to_csv(OUTPUT_DIR / 'recommendation_10pct_per_class.csv', index=False)

    cm = confusion_matrix(test_df['label_id'], prediction_ids, labels=list(ID2LABEL))
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, matrix, title, value_format in [
        (axes[0], cm, '10% model — counts', 'd'),
        (axes[1], cm_normalized, '10% model — row normalized', '.2f'),
    ]:
        image = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks(range(3), VALID_LABELS, rotation=20)
        ax.set_yticks(range(3), VALID_LABELS)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        threshold = matrix.max() / 2
        for row in range(3):
            for column in range(3):
                ax.text(column, row, format(matrix[row, column], value_format), ha='center', va='center',
                        color='white' if matrix[row, column] > threshold else 'black')
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'recommendation_10pct_confusion_matrix.png', dpi=180, bbox_inches='tight')
    plt.show()

    (OUTPUT_DIR / 'recommendation_10pct_bootstrap_results.json').write_text(
        json.dumps(bootstrap, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    (OUTPUT_DIR / 'recommendation_10pct_latency_results.json').write_text(
        json.dumps(latency, ensure_ascii=False, indent=2), encoding='utf-8',
    )

    # slice/error artifacts برای بخش چهارم ارزیابی پروژه.
    token_lengths = []
    for start in range(0, len(test_df), 512):
        encoded = tokenizer(
            test_df['text_full'].iloc[start:start + 512].tolist(),
            truncation=False, padding=False, return_length=True, verbose=False,
        )
        token_lengths.extend(encoded['length'])
    analysis_frame = test_df[['id', 'text_group_id', 'recommendation_status', 'title', 'body', 'advantages', 'disadvantages']].copy()
    analysis_frame['prediction'] = predictions['prediction_10pct'].to_numpy()
    analysis_frame['prediction_id'] = prediction_ids
    analysis_frame['label_id'] = test_df['label_id'].to_numpy()
    analysis_frame['confidence_score'] = scores.max(axis=1)
    analysis_frame['token_length'] = np.asarray(token_lengths, dtype=int)
    analysis_frame['was_truncated'] = analysis_frame['token_length'].gt(MAX_LENGTH)
    analysis_frame['has_title'] = analysis_frame['title'].ne('')
    analysis_frame['has_structured_pros_or_cons'] = analysis_frame['advantages'].ne('') | analysis_frame['disadvantages'].ne('')
    analysis_frame['length_bucket'] = pd.cut(
        analysis_frame['token_length'], bins=[0, 16, 48, 96, MAX_LENGTH, np.inf],
        labels=['very_short_1_16', 'short_17_48', 'medium_49_96', f'long_97_{MAX_LENGTH}', f'truncated_gt_{MAX_LENGTH}'],
        include_lowest=True,
    ).astype(str)
    slice_masks = {
        'all_test': np.ones(len(analysis_frame), dtype=bool),
        'title_missing': ~analysis_frame['has_title'],
        'title_present': analysis_frame['has_title'],
        'structured_pros_or_cons_present': analysis_frame['has_structured_pros_or_cons'],
        'structured_pros_or_cons_missing': ~analysis_frame['has_structured_pros_or_cons'],
        'truncated': analysis_frame['was_truncated'],
        'not_truncated': ~analysis_frame['was_truncated'],
    }
    for bucket in analysis_frame['length_bucket'].dropna().unique():
        slice_masks[f'length::{bucket}'] = analysis_frame['length_bucket'].eq(bucket)
    slice_rows = []
    for slice_name, mask in slice_masks.items():
        subset = analysis_frame.loc[np.asarray(mask)]
        if not subset.empty:
            slice_rows.append({
                'slice': slice_name, 'rows': len(subset),
                'groups': int(subset['text_group_id'].nunique()),
                **calculate_metrics(subset['label_id'], subset['prediction_id']),
            })
    slice_results = pd.DataFrame(slice_rows)
    slice_results.to_csv(OUTPUT_DIR / 'recommendation_10pct_slice_results.csv', index=False)
    failures = analysis_frame[analysis_frame['label_id'].ne(analysis_frame['prediction_id'])].copy()
    failures['confusion_pair'] = failures['recommendation_status'] + ' -> ' + failures['prediction']
    failures.to_csv(OUTPUT_DIR / 'recommendation_10pct_failure_cases.csv', index=False)

    integration_contract = {
        'schema_version': '1.0.0',
        'model_version': MODEL_RELEASE_VERSION if release_passed else MODEL_RC_VERSION,
        'status': 'RELEASED' if release_passed else 'EXPERIMENT',
        'operation': 'predict_recommendation_status',
        'input': {'title': 'string|null', 'body': 'string|null', 'advantages': 'string|null', 'disadvantages': 'string|null'},
        'output': {
            'label': VALID_LABELS,
            'scores': {label: 'uncalibrated float model score' for label in VALID_LABELS},
        },
        'preprocessing_version': PREPROCESSING_VERSION,
        'max_length': MAX_LENGTH,
    }
    (OUTPUT_DIR / 'recommendation_10pct_integration_contract.json').write_text(
        json.dumps(integration_contract, ensure_ascii=False, indent=2), encoding='utf-8',
    )

    def directory_artifact_sha256(directory: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in directory.rglob('*') if item.is_file()):
            digest.update(path.relative_to(directory).as_posix().encode('utf-8'))
            digest.update(bytes.fromhex(file_sha256(path)))
        return digest.hexdigest()

    model_artifact_sha256 = directory_artifact_sha256(release_dir)
    prediction_path = OUTPUT_DIR / 'recommendation_10pct_test_predictions.csv'
    prediction_sha256 = file_sha256(prediction_path)
    release_card = f'''# Digikala recommendation classifier — 10% candidate

- Decision: {'PASS — promote v2' if release_passed else 'FAIL — keep v1'}
- Model version: {MODEL_RELEASE_VERSION if release_passed else MODEL_RC_VERSION}
- Test Macro-F1: {metrics['macro_f1']:.6f}
- Reproduced v1 Macro-F1: {reproduced_v1_metrics['macro_f1']:.6f}
- Absolute gain: {metrics['macro_f1'] - reproduced_v1_metrics['macro_f1']:+.6f}
- Paired delta CI95: {bootstrap['paired_macro_f1_delta_ci95']}
- Required gates: {gates}
- Advisory checks: {advisory_checks}
- Model SHA-256: {model_artifact_sha256}
'''
    (OUTPUT_DIR / 'recommendation_10pct_release_card.md').write_text(release_card, encoding='utf-8')

    final_summary = {
        'task': 'digikala_recommendation_status_10pct_final_evaluation',
        'decision': 'PASS_PROMOTE_V2' if release_passed else 'FAIL_KEEP_V1',
        'release_passed': release_passed,
        'model_version': MODEL_RELEASE_VERSION if release_passed else MODEL_RC_VERSION,
        'source_sha256': HF_EXPECTED_SHA256,
        'expanded_manifest_digest': manifest_digest,
        'model': MODEL_NAME,
        'model_revision': MODEL_REVISION,
        'max_length': MAX_LENGTH,
        'preprocessing': PREPROCESSING_VERSION,
        'selected_epochs': selected_epochs,
        'validation_summary_source': prior_validation_source,
        'v1_predictions_source': v1_predictions_source,
        'v1_predictions_sha256': v1_predictions_sha256,
        'v1_reference_metrics': V1_TEST,
        'v1_reproduced_metrics': reproduced_v1_metrics,
        'v1_reproduction_delta': v1_reproduction_delta,
        'test_metrics_10pct': metrics,
        'test_confusion_matrix_label_order': VALID_LABELS,
        'test_confusion_matrix': cm.tolist(),
        'absolute_macro_f1_gain': metrics['macro_f1'] - reproduced_v1_metrics['macro_f1'],
        'bootstrap': bootstrap,
        'latency': latency,
        'gates': gates,
        'advisory_checks': advisory_checks,
        'fit_seconds_this_session': final_fit_seconds,
        'test_predict_seconds': test_predict_seconds,
        'trainer_train_metrics': trainer_train_metrics,
        'sampling_audit': sampling_audit,
        'runtime': runtime_info,
        'model_dir': str(release_dir),
        'model_artifact_sha256': model_artifact_sha256,
        'test_predictions_sha256': prediction_sha256,
        'slice_count': len(slice_results),
        'failure_rows': len(failures),
    }
    final_summary_path = OUTPUT_DIR / 'recommendation_10pct_evaluation_summary.json'
    final_summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding='utf-8')

    manifest_entries = []
    manifest_targets = [
        final_summary_path, prediction_path,
        OUTPUT_DIR / 'recommendation_10pct_per_class.csv',
        OUTPUT_DIR / 'recommendation_10pct_slice_results.csv',
        OUTPUT_DIR / 'recommendation_10pct_failure_cases.csv',
        OUTPUT_DIR / 'recommendation_10pct_confusion_matrix.png',
        OUTPUT_DIR / 'recommendation_10pct_bootstrap_results.json',
        OUTPUT_DIR / 'recommendation_10pct_latency_results.json',
        OUTPUT_DIR / 'recommendation_10pct_integration_contract.json',
        OUTPUT_DIR / 'recommendation_10pct_release_card.md',
    ]
    for path in manifest_targets:
        manifest_entries.append(f'{file_sha256(path)}  {path.name}')
    (OUTPUT_DIR / 'MANIFEST_10PCT.sha256').write_text('\n'.join(manifest_entries) + '\n', encoding='utf-8')

    print('\n' + '#' * 24 + ' COPY THIS FINAL SUMMARY ' + '#' * 24)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print('\nRelease action:', 'Build/publish v2 without deleting v1.' if release_passed else 'Keep public v1; retain this run only as an experiment.')
else:
    print('RUN_PHASE=validation: final/test cell skipped. Test predictions were not computed.')

# %% [markdown]
# ## خروجی‌هایی که باید نگه دارید
#
# بعد از فاز validation:
#
# - `recommendation_10pct_validation_summary.json`
# - `recommendation_10pct_validation_predictions.csv`
# - `recommendation_10pct_split_manifest.csv`
# - `recommendation_10pct_data_audit.json`
# - `best_transformer_encoder_10pct_validation/`
# - پوشهٔ run/checkpoint برای resume (فقط نسخهٔ میانی)
#
# بعد از فاز final:
#
# - `recommendation_10pct_evaluation_summary.json`
# - `recommendation_10pct_test_predictions.csv`
# - `recommendation_10pct_bootstrap_results.json`
# - `recommendation_10pct_latency_results.json`
# - `recommendation_10pct_slice_results.csv`
# - `recommendation_10pct_failure_cases.csv`
# - `recommendation_10pct_confusion_matrix.png`
# - `recommendation_10pct_integration_contract.json`
# - `recommendation_10pct_release_card.md`
# - `MANIFEST_10PCT.sha256`
# - `best_transformer_encoder_10pct/`
#
# checkpointها را داخل Dataset انتشار نهایی نگذارید. آن‌ها فقط برای resume هستند. بلوک
# `COPY THIS ... SUMMARY` را پس از هر فاز برای بررسی بفرستید.
