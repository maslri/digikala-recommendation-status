# %% [markdown]
# # Build the v2 release Dataset without retraining
#
# This notebook takes the saved output of the final 10% model run as Input and copies only
# the files needed for release into a clean directory. It performs no download, training, or inference.
# The notebook output contains only the following directory:
#
# `digikala_recommendation_status_xlm_roberta_10pct_v2/`
#
# After completion, create a new Kaggle Dataset from this notebook's complete Output.

# %% [markdown]
# ## Before running
#
# 1. In the seven-hour notebook, preserve current outputs with **Quick Save** and the output-saving option;
#    `Save & Run All` is not required.
# 2. Create a new notebook with Accelerator `None` and Internet `Off`.
# 3. Attach the final run's Saved Output to the new notebook with `Add Input`.
# 4. Run this notebook. It normally takes only the time needed to copy the approximately 1.1 GB model.

# %%
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


INPUT_ROOT = Path('/kaggle/input')
WORKING_ROOT = Path('/kaggle/working')
RELEASE_NAME = 'digikala_recommendation_status_xlm_roberta_10pct_v2'
OUTPUT_ROOT = WORKING_ROOT / RELEASE_NAME

# If auto-discovery finds multiple valid final runs, set the correct output root here.
# Example: '/kaggle/input/notebooks/maslri/my-final-notebook'
SOURCE_ROOT_OVERRIDE = ''

EXPECTED = {
    'task': 'digikala_recommendation_status_10pct_final_evaluation',
    'decision': 'PASS_PROMOTE_V2',
    'release_passed': True,
    'model_version': 'digikala-rec-xlm-roberta-10pct-v2.0.0',
    'source_sha256': 'c7a8aa3020334fde8ec24944576a03fe5785e6fe12cd01042f5836632ddf8297',
    'model': 'FacebookAI/xlm-roberta-base',
    'model_revision': 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089',
    'max_length': 128,
    'preprocessing': 'fa_light_v1',
    'model_artifact_sha256': 'bc99ce62255893faf29fa46c27b3faa798b59fd09d4d31b5ab5ee280df8c570d',
    'test_predictions_sha256': 'f20eabe60134c77c57c6020c36a5db64fdbeb3d952e603178ccc4345e4f7ea57',
}

SUMMARY_NAME = 'recommendation_10pct_evaluation_summary.json'
SOURCE_MODEL_NAME = 'best_transformer_encoder_10pct'

REQUIRED_ARTIFACTS = {
    'evaluation': [
        SUMMARY_NAME,
        'recommendation_10pct_test_predictions.csv',
        'recommendation_10pct_per_class.csv',
        'recommendation_10pct_slice_results.csv',
        'recommendation_10pct_failure_cases.csv',
        'recommendation_10pct_confusion_matrix.png',
        'recommendation_10pct_bootstrap_results.json',
        'recommendation_10pct_latency_results.json',
        'recommendation_10pct_release_card.md',
    ],
    'metadata': [
        'recommendation_10pct_integration_contract.json',
        'MANIFEST_10PCT.sha256',
    ],
}

# These artifacts help auditing and reproduction, but their absence does not invalidate inference.
OPTIONAL_ARTIFACTS = {
    'metadata': [
        'recommendation_10pct_split_manifest.csv',
        'recommendation_10pct_data_audit.json',
        'recommendation_10pct_training_history.csv',
    ],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def directory_artifact_sha256(directory: Path) -> str:
    # Use exactly the same model-hash algorithm as the final notebook.
    digest = hashlib.sha256()
    paths = sorted(item for item in directory.rglob('*') if item.is_file())
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode('utf-8'))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def validate_summary(summary: dict, path: Path) -> None:
    errors = []
    for key, expected_value in EXPECTED.items():
        actual_value = summary.get(key)
        if actual_value != expected_value:
            errors.append(f'{key}: expected={expected_value!r}, actual={actual_value!r}')
    if errors:
        raise RuntimeError(
            f'Final summary is not the approved v2 release: {path}\n- ' + '\n- '.join(errors)
        )


def discover_source_root() -> tuple[Path, dict, Path]:
    if SOURCE_ROOT_OVERRIDE.strip():
        source_root = Path(SOURCE_ROOT_OVERRIDE.strip())
        summary_path = source_root / SUMMARY_NAME
        if not summary_path.is_file():
            raise FileNotFoundError(f'Missing summary under SOURCE_ROOT_OVERRIDE: {summary_path}')
        summary = read_json(summary_path)
        validate_summary(summary, summary_path)
        model_dir = source_root / SOURCE_MODEL_NAME
        if not model_dir.is_dir():
            raise FileNotFoundError(f'Missing model directory: {model_dir}')
        return source_root, summary, summary_path

    candidates = []
    for summary_path in INPUT_ROOT.rglob(SUMMARY_NAME):
        try:
            summary = read_json(summary_path)
            validate_summary(summary, summary_path)
        except Exception:
            continue
        source_root = summary_path.parent
        if (source_root / SOURCE_MODEL_NAME).is_dir():
            candidates.append((source_root, summary, summary_path))

    unique = {}
    for source_root, summary, summary_path in candidates:
        unique[str(source_root.resolve())] = (source_root, summary, summary_path)
    candidates = list(unique.values())

    if not candidates:
        summaries = [str(path) for path in INPUT_ROOT.rglob(SUMMARY_NAME)]
        raise FileNotFoundError(
            'Approved final v2 output was not found under /kaggle/input. '
            'Add the saved final Notebook Output as Input. Found summaries: '
            + repr(summaries[:20])
        )
    if len(candidates) > 1:
        roots = [str(item[0]) for item in candidates]
        raise RuntimeError(
            'More than one approved final output was found. Set SOURCE_ROOT_OVERRIDE to one root: '
            + repr(roots)
        )
    return candidates[0]


source_root, final_summary, summary_path = discover_source_root()
source_model_dir = source_root / SOURCE_MODEL_NAME

print('Approved source root:', source_root)
print('Final summary:', summary_path)
print('Source model:', source_model_dir)

# Confirm that no required file is missing before copying starts.
missing_required = []
for names in REQUIRED_ARTIFACTS.values():
    missing_required.extend(name for name in names if not (source_root / name).is_file())
if missing_required:
    raise FileNotFoundError(f'Missing required final artifacts: {missing_required}')

source_prediction = source_root / 'recommendation_10pct_test_predictions.csv'
source_prediction_sha = file_sha256(source_prediction)
if source_prediction_sha != EXPECTED['test_predictions_sha256']:
    raise RuntimeError(
        f'Test prediction SHA-256 mismatch: {source_prediction_sha} '
        f'!= {EXPECTED["test_predictions_sha256"]}'
    )

# Pre-copy model hashing takes a few minutes but prevents publishing the wrong directory.
source_model_sha = directory_artifact_sha256(source_model_dir)
if source_model_sha != EXPECTED['model_artifact_sha256']:
    raise RuntimeError(
        f'Model artifact SHA-256 mismatch: {source_model_sha} '
        f'!= {EXPECTED["model_artifact_sha256"]}'
    )

print('Source hashes: PASS')

# %% [markdown]
# ## Build the clean release directory
#
# A ZIP is not created because keeping it beside the model directory nearly doubles output size.

# %%
if OUTPUT_ROOT.exists():
    raise RuntimeError(
        f'Output already exists: {OUTPUT_ROOT}. Restart the packaging session before rerunning.'
    )

(OUTPUT_ROOT / 'evaluation').mkdir(parents=True)
(OUTPUT_ROOT / 'metadata').mkdir(parents=True)

shutil.copytree(source_model_dir, OUTPUT_ROOT / 'model')

copied_artifacts = []
for destination_group, names in REQUIRED_ARTIFACTS.items():
    for name in names:
        destination_name = 'SOURCE_MANIFEST_10PCT.sha256' if name == 'MANIFEST_10PCT.sha256' else name
        destination = OUTPUT_ROOT / destination_group / destination_name
        shutil.copy2(source_root / name, destination)
        copied_artifacts.append(str(destination.relative_to(OUTPUT_ROOT)))

optional_copied = []
for destination_group, names in OPTIONAL_ARTIFACTS.items():
    for name in names:
        source = source_root / name
        if source.is_file():
            destination = OUTPUT_ROOT / destination_group / name
            shutil.copy2(source, destination)
            optional_copied.append(str(destination.relative_to(OUTPUT_ROOT)))

readme = f'''# Digikala Recommendation Status — XLM-RoBERTa 10% v2

Production artifact for project section 3: three-class purchase-recommendation prediction.

## Release

- Decision: `{final_summary['decision']}`
- Model version: `{final_summary['model_version']}`
- Base model: `{final_summary['model']}`
- Test Macro-F1: `{final_summary['test_metrics_10pct']['macro_f1']:.6f}`
- Macro-F1 gain over v1: `{final_summary['absolute_macro_f1_gain']:+.6f}`
- Labels, in model order: `recommended`, `not_recommended`, `no_idea`
- Max token length: `{final_summary['max_length']}`
- Preprocessing: `{final_summary['preprocessing']}`

## Layout

- `model/`: Hugging Face model, tokenizer and `inference_config.json`
- `evaluation/`: locked-test metrics, predictions, errors, slices, bootstrap and latency results
- `metadata/`: integration contract, data/split audit and source manifest when available
- `RELEASE_MANIFEST.sha256`: SHA-256 for every packaged file except the manifest itself

## Load in Kaggle

```python
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

root = Path('/kaggle/input/<dataset-slug>/{RELEASE_NAME}')
tokenizer = AutoTokenizer.from_pretrained(root / 'model', local_files_only=True, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(root / 'model', local_files_only=True)
```

Input fields and output schema are defined in
`metadata/recommendation_10pct_integration_contract.json`. Apply the exact `fa_light_v1`
normalizer used by the project integration code before tokenization. Softmax values are
uncalibrated model scores, not guaranteed probabilities.

The previous public v1 dataset must remain available for rollback and comparison.
'''
(OUTPUT_ROOT / 'README.md').write_text(readme, encoding='utf-8')

release_summary = {
    'decision': 'PASS',
    'release_source_decision': final_summary['decision'],
    'model_version': final_summary['model_version'],
    'model_artifact_sha256': source_model_sha,
    'test_predictions_sha256': source_prediction_sha,
    'source_dataset_sha256': final_summary['source_sha256'],
    'model_revision': final_summary['model_revision'],
    'test_macro_f1': final_summary['test_metrics_10pct']['macro_f1'],
    'absolute_macro_f1_gain_over_v1': final_summary['absolute_macro_f1_gain'],
    'packaged_at_utc': datetime.now(timezone.utc).isoformat(),
    'required_artifacts': copied_artifacts,
    'optional_artifacts_copied': optional_copied,
}
(OUTPUT_ROOT / 'release_dataset_summary.json').write_text(
    json.dumps(release_summary, ensure_ascii=False, indent=2), encoding='utf-8'
)

# %% [markdown]
# ## Final package validation

# %%
copied_model_sha = directory_artifact_sha256(OUTPUT_ROOT / 'model')
copied_prediction_sha = file_sha256(
    OUTPUT_ROOT / 'evaluation' / 'recommendation_10pct_test_predictions.csv'
)
if copied_model_sha != source_model_sha:
    raise RuntimeError('Copied model hash does not match the approved source model.')
if copied_prediction_sha != source_prediction_sha:
    raise RuntimeError('Copied prediction hash does not match the approved source predictions.')

config = read_json(OUTPUT_ROOT / 'model' / 'config.json')
inference_config = read_json(OUTPUT_ROOT / 'model' / 'inference_config.json')
expected_labels = ['recommended', 'not_recommended', 'no_idea']

# `num_labels` is a derived Transformers property; some versions do not serialize it as
# a standalone config.json key. In that case, validate the actual head order with id2label
# and inference_config.
config_num_labels = config.get('num_labels')
if config_num_labels is not None and int(config_num_labels) != len(expected_labels):
    raise RuntimeError(f'Unexpected num_labels in model config: {config_num_labels}')

config_id2label = config.get('id2label', {})
if config_id2label:
    config_label_order = [
        config_id2label.get(str(index), config_id2label.get(index))
        for index in range(len(expected_labels))
    ]
    if config_label_order != expected_labels:
        raise RuntimeError(f'Unexpected id2label order in model config: {config_label_order}')
if inference_config.get('model_version') != EXPECTED['model_version']:
    raise RuntimeError(f'Unexpected inference model version: {inference_config}')
if inference_config.get('labels') != expected_labels:
    raise RuntimeError(f'Unexpected label order: {inference_config.get("labels")}')

manifest_lines = []
for path in sorted(item for item in OUTPUT_ROOT.rglob('*') if item.is_file()):
    if path.name == 'RELEASE_MANIFEST.sha256':
        continue
    manifest_lines.append(f'{file_sha256(path)}  {path.relative_to(OUTPUT_ROOT).as_posix()}')
(OUTPUT_ROOT / 'RELEASE_MANIFEST.sha256').write_text(
    '\n'.join(manifest_lines) + '\n', encoding='utf-8'
)

file_count = sum(1 for path in OUTPUT_ROOT.rglob('*') if path.is_file())
package_size_bytes = sum(path.stat().st_size for path in OUTPUT_ROOT.rglob('*') if path.is_file())

package_summary = {
    'decision': 'PASS',
    'model_release': EXPECTED['model_version'],
    'source_root': str(source_root),
    'release_folder': str(OUTPUT_ROOT),
    'model_artifact_sha256': copied_model_sha,
    'test_predictions_sha256': copied_prediction_sha,
    'release_file_count': file_count,
    'package_size_bytes': package_size_bytes,
    'zip_created': False,
    'training_or_inference_ran': False,
}

print('\n' + '#' * 20 + ' COPY THIS PACKAGE SUMMARY ' + '#' * 20)
print(json.dumps(package_summary, ensure_ascii=False, indent=2))
print('\nNext action: Save Version, then create a new Kaggle Dataset from this Notebook Output.')

# %% [markdown]
# ## Publish on Kaggle
#
# When the summary above reports `PASS`:
#
# 1. `Save Version` for this short notebook; rerunning it performs no training.
# 2. Create a new Dataset from this version's Output.
# 3. Suggested slug: `digikala-recommendation-status-xlm-roberta-10pct-v2`
# 4. Create the Dataset as Private, verify its README and size, then make it Public.
# 5. Do not delete or overwrite the public v1 Dataset.
