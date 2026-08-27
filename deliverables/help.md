# راهنمای استفاده از مدل پیش‌بینی وضعیت پیشنهاد خرید

## معرفی

این مؤلفه مربوط به بخش سوم پروژه، «پیش‌بینی وضعیت پیشنهاد خرید»، است. مدل نهایی با fine-tune کردن `XLM-RoBERTa-base` ساخته شده و نظرهای فارسی دیجی‌کالا را در سه کلاس زیر طبقه‌بندی می‌کند:

```text
recommended
not_recommended
no_idea
```

مدل، tokenizer، کد inference، قرارداد اتصال، گزارش‌ها و نتایج ارزیابی در Dataset عمومی Kaggle زیر قرار دارند:

https://www.kaggle.com/datasets/maslri/digikala-recommendation-status-xlm-roberta-v1/data

## جایگاه این مؤلفه در پروژه نهایی

خروجی کلی پروژه صرفاً یک LLM واحد نیست. پروژه نهایی یک سیستم چندبخشی است که می‌تواند در ظاهر یک دستیار گفت‌وگویی فارسی باشد، اما در پشت صحنه از چند مؤلفه مستقل استفاده می‌کند:

```text
کاربر
  │
  ▼
رابط گفتگو یا API
  │
  ▼
LLM / Orchestrator
  │
  ├── جست‌وجو و کشف محصول
  ├── بازیابی نظرات و شواهد
  ├── پرسش و پاسخ درباره محصول
  ├── مقایسه محصولات
  ├── تحلیل مدیریتی
  └── این مؤلفه: پیش‌بینی recommendation_status
  │
  ▼
ترکیب اطلاعات محصول، شواهد، آمار و پیش‌بینی‌ها
  │
  ▼
پاسخ نهایی فارسی به کاربر
```

LLM وظیفه فهم سؤال و تولید پاسخ طبیعی را دارد. این مدل وظیفه‌ای محدود و قابل‌اندازه‌گیری دارد: دریافت متن یک نظر و تعیین یکی از سه وضعیت `recommended`، `not_recommended` یا `no_idea`.

وزن‌های این مدل نباید با وزن‌های LLM ترکیب شوند و نیازی به fine-tune کردن مجدد LLM وجود ندارد. ادغام در سطح کد انجام می‌شود: orchestrator مدل را مانند یک ابزار یا سرویس فراخوانی می‌کند و خروجی ساختاریافته آن را در اختیار بخش‌های دیگر قرار می‌دهد.

## کاربرد خروجی مدل در سایر بخش‌ها

### تکمیل نظرهای فاقد برچسب

اگر `recommendation_status` واقعی موجود باشد، همان مقدار استفاده می‌شود. مدل فقط برای نظر جدید یا نظر فاقد برچسب فراخوانی می‌شود.

### محاسبه رضایت کاربران

برچسب‌های واقعی و پیش‌بینی‌شده را می‌توان در سطح محصول تجمیع کرد:

```json
{
  "recommended_percent": 72.4,
  "not_recommended_percent": 14.1,
  "no_idea_percent": 13.5
}
```

این آمار می‌تواند به LLM داده شود تا درباره میزان رضایت کاربران توضیح دهد. این درصدها با score اطمینان مدل تفاوت دارند: درصدهای بالا از شمارش برچسب نظرها محاسبه می‌شوند، درحالی‌که `scores` خروجی مدل برای یک نظر کالیبره نیستند.

### جست‌وجو و رتبه‌بندی محصول

برای پرسش‌هایی مانند «محصولی معرفی کن که خریداران از آن راضی باشند»، درصد نظرهای `recommended` می‌تواند یکی از سیگنال‌های رتبه‌بندی یا فیلتر محصولات باشد.

### مقایسه محصولات

در مقایسه دو محصول، سیستم می‌تواند آمار سه کلاس، نظرهای شاهد، قیمت و ویژگی‌های محصول را کنار هم قرار دهد و LLM براساس این داده ساختاریافته پاسخ نهایی را تولید کند.

### تحلیل مدیریتی

برای پرسش‌هایی مانند «کدام محصولات نظر زیادی دارند ولی درصد پیشنهاد خریدشان پایین است؟»، خروجی این مدل امکان ساخت شاخص‌های تجمیعی در سطح محصول، برند یا دسته را فراهم می‌کند.

## دو روش استفاده در سیستم نهایی

### روش اول: پردازش آفلاین

برای تعداد زیادی نظر فاقد برچسب، مدل یک بار به‌صورت batch اجرا می‌شود و خروجی در جدول نظرات ذخیره می‌شود. این روش برای جست‌وجو، مقایسه، داشبورد و تحلیل مدیریتی مناسب‌تر و کم‌هزینه‌تر است.

### روش دوم: پیش‌بینی آنلاین

برای یک نظر جدید، orchestrator متد `predict_one` یا `resolve` را هنگام درخواست فراخوانی می‌کند. این روش برای ورودی‌های جدید یا Demo تعاملی مناسب است.

در سیستم نهایی می‌توان هر دو روش را هم‌زمان داشت: داده‌های قدیمی به‌صورت آفلاین پردازش شوند و نظرهای جدید به‌صورت آنلاین پیش‌بینی شوند.

## اطلاعات نسخه

```text
Model release: digikala-rec-xlm-roberta-2pct-v1.0.0
Preprocessing: fa_light_v1
Maximum length: 128 tokens
Release decision: PASS
Smoke test: PASS
```

اثر انگشت SHA-256 وزن مدل:

```text
f343a3bcee07c68beaabece504a9efd1f200661e376f0b5235c75c7c9c394cf4
```

## نتایج ارزیابی

| معیار | مقدار |
|---|---:|
| Macro-F1 | 0.7172 |
| Weighted-F1 | 0.8554 |
| Accuracy | 0.8423 |
| Baseline Macro-F1 | 0.6611 |
| بهبود مطلق Macro-F1 | 0.0560 |

این نسخه روی نمونه ثابت دو درصدی داده، شامل 105,297 ردیف، آموزش داده شده است. آموزش روی ده درصد داده به‌عنوان کار آینده در نظر گرفته شده و جزو این نسخه نیست.

## اضافه‌کردن مدل به Notebook در Kaggle

1. در Notebook مقصد، پنل `Input` را باز کنید.
2. روی `Add Input` بزنید.
3. عبارت زیر را جست‌وجو کنید:

```text
digikala-recommendation-status-xlm-roberta-v1
```

4. Dataset متعلق به کاربر `maslri` را اضافه کنید.

## بارگذاری مدل

کد زیر محل بسته را بدون وابستگی به slug دقیق مسیر Kaggle پیدا می‌کند:

```python
from pathlib import Path
import sys

package_candidates = [
    path
    for path in Path("/kaggle/input").rglob("recommendation_prediction_v1")
    if (
        path
        / "src"
        / "recommendation_prediction"
        / "predictor.py"
    ).is_file()
]

if len(package_candidates) != 1:
    raise RuntimeError(
        f"Expected exactly one recommendation package, found: {package_candidates}"
    )

package_root = package_candidates[0]
model_dir = package_root / "model" / "best_transformer_encoder"

sys.path.insert(0, str(package_root / "src"))

from recommendation_prediction import RecommendationPredictor

predictor = RecommendationPredictor(model_dir)

print("Package:", package_root)
print("Model:", model_dir)
print(predictor.health_check())
```

در صورت موفقیت، `health_check()` باید مقدار `status="ok"` برگرداند.

## پیش‌بینی یک نظر جدید

```python
result = predictor.predict_one(
    comment_id="123",
    product_id="456",
    title="عالی بود",
    body="از کیفیت این محصول خیلی راضی هستم",
    advantages="کیفیت خوب و قیمت مناسب",
    disadvantages="",
)

result
```

## حفظ برچسب واقعی موجود

اگر ممکن است رکورد از قبل دارای `recommendation_status` واقعی باشد، به‌جای `predict_one` از `resolve` استفاده کنید:

```python
record = {
    "comment_id": "123",
    "product_id": "456",
    "title": "عالی بود",
    "body": "از کیفیت این محصول خیلی راضی هستم",
    "advantages": "",
    "disadvantages": "",
    "recommendation_status": None,
}

result = predictor.resolve(record)
```

رفتار `resolve`:

- اگر `recommendation_status` یکی از سه برچسب معتبر باشد، همان مقدار حفظ می‌شود و `source` برابر `observed` خواهد بود.
- اگر برچسب وجود نداشته باشد، مدل پیش‌بینی می‌کند و `source` برابر `model_prediction` خواهد بود.

## پیش‌بینی دسته‌ای

برای تعداد زیادی نظر از `predict_batch` استفاده کنید:

```python
records = [
    {
        "comment_id": "1",
        "product_id": "100",
        "title": "عالی",
        "body": "کاملاً از خرید راضی هستم",
        "advantages": "کیفیت مناسب",
        "disadvantages": "",
    },
    {
        "comment_id": "2",
        "product_id": "101",
        "title": "نخرید",
        "body": "کیفیت بسیار ضعیفی داشت و مرجوع کردم",
        "advantages": "",
        "disadvantages": "کیفیت پایین",
    },
]

results = predictor.predict_batch(records, batch_size=32)
```

## قرارداد ورودی

| فیلد | نوع | توضیح |
|---|---|---|
| `comment_id` | `string` یا `null` | شناسه اختیاری نظر |
| `product_id` | `string` یا `null` | شناسه اختیاری محصول |
| `title` | `string` | عنوان نظر |
| `body` | `string` | متن اصلی نظر |
| `advantages` | `string` | مزایای ثبت‌شده |
| `disadvantages` | `string` | معایب ثبت‌شده |
| `recommendation_status` | برچسب یا `null` | فقط برای متد `resolve` |

در نسخه اول قرارداد، چهار فیلد متنی باید رشته باشند. برای `advantages` و `disadvantages` آرایه JSON ارسال نکنید.

اگر هر چهار فیلد `title`، `body`، `advantages` و `disadvantages` خالی باشند، درخواست با خطا رد می‌شود.

## ساختار خروجی

```json
{
  "component": "recommendation_prediction",
  "schema_version": "1.0.0",
  "comment_id": "123",
  "product_id": "456",
  "label": "recommended",
  "scores": {
    "recommended": 0.91,
    "not_recommended": 0.03,
    "no_idea": 0.06
  },
  "confidence_score": 0.91,
  "score_margin": 0.85,
  "scores_are_calibrated_probabilities": false,
  "model_version": "digikala-rec-xlm-roberta-2pct-v1.0.0",
  "artifact_sha256": "f343a3bcee07c68beaabece504a9efd1f200661e376f0b5235c75c7c9c394cf4",
  "preprocessing_version": "fa_light_v1",
  "source": "model_prediction",
  "latency_ms": 11.4
}
```

## ترتیب ثابت کلاس‌ها

ترتیب شناسه کلاس‌ها بخشی از قرارداد مدل است و نباید تغییر کند:

```text
0 = recommended
1 = not_recommended
2 = no_idea
```

## نکات مهم ادغام

- بخش‌های دیگر سیستم باید مقدار `label` را به‌عنوان خروجی اصلی مصرف کنند.
- امتیازهای `scores` خروجی Softmax کالیبره‌نشده‌اند و احتمال واقعی محسوب نمی‌شوند.
- برای تصمیم‌های حساس یا نمایش درصد اطمینان قطعی از score خام استفاده نکنید.
- اگر برچسب واقعی وجود دارد، آن را با prediction جایگزین نکنید.
- preprocessing، ترتیب کلاس‌ها و `max_length` نباید در بخش‌های دیگر دوباره پیاده‌سازی یا تغییر داده شوند؛ از ماژول تحویلی استفاده کنید.
- ورودی‌ها باید متن خام باشند و بخش دیگر سیستم نباید تگ‌های `[TITLE]` یا `[BODY]` را دستی اضافه کند.
- برای inference آفلاین، اتصال اینترنت لازم نیست.

## اتصال به بخش‌های LLM

این مدل یک مؤلفه مستقل NLP است. LLM نباید متن نظر را دوباره برای حدس‌زدن `recommendation_status` تحلیل کند؛ این مسئولیت به مؤلفه حاضر سپرده می‌شود. LLM نتیجه مدل، آمار تجمیعی و شواهد بازیابی‌شده را دریافت کرده و فقط پاسخ نهایی را تولید می‌کند.

جریان پیشنهادی برای یک پرسش درباره رضایت کاربران:

```text
سؤال کاربر
  ↓
تشخیص محصول توسط LLM یا Router
  ↓
بازیابی محصول و نظرات مرتبط
  ↓
استفاده از recommendation_status واقعی یا پیش‌بینی مدل
  ↓
محاسبه درصد هر کلاس و انتخاب نظرهای شاهد
  ↓
ارسال آمار و شناسه شواهد به LLM
  ↓
تولید پاسخ مستند فارسی
```

نمونه داده‌ای که می‌تواند به LLM داده شود:

```json
{
  "product": {
    "product_id": "456",
    "title": "نام محصول",
    "price": 1200000
  },
  "recommendation_analysis": {
    "total_comments": 250,
    "recommended_percent": 72.4,
    "not_recommended_percent": 14.1,
    "no_idea_percent": 13.5,
    "model_version": "digikala-rec-xlm-roberta-2pct-v1.0.0"
  },
  "evidence_comment_ids": ["123", "456", "789"]
}
```

کلید پیشنهادی برای نتیجه یک نظر در خروجی تجمیع‌شده سیستم:

```json
{
  "recommendation_prediction": {
    "label": "recommended",
    "source": "model_prediction",
    "model_version": "digikala-rec-xlm-roberta-2pct-v1.0.0"
  }
}
```

حداقل ادغام قابل‌قبول این است که این مؤلفه به‌صورت یک ماژول مستقل کنار سیستم اصلی وجود داشته باشد و ارزیابی آن گزارش شود. حالت پیشنهادی بهتر این است که بخش‌های جست‌وجو، مقایسه، پرسش و پاسخ و تحلیل مدیریتی از آمار حاصل از آن استفاده کنند.

## فایل‌های مهم داخل Dataset

```text
recommendation_prediction_v1/
├── model/best_transformer_encoder/
├── src/recommendation_prediction/
├── config/integration_contract.json
├── artifacts/training/
├── artifacts/evaluation/
├── reports/
├── notebooks/
├── requirements.txt
└── MANIFEST.sha256
```

قرارداد کامل اتصال:

```text
recommendation_prediction_v1/config/integration_contract.json
```

گزارش نهایی بخش سوم و چهارم:

```text
recommendation_prediction_v1/reports/section3_and_4_final_report_fa.md
```

## محدودیت‌های نسخه فعلی

- مدل روی نمونه ثابت دو درصدی داده آموزش دیده است.
- کلاس `no_idea` همچنان ضعیف‌ترین کلاس است.
- مدل ممکن است در متن‌های کنایه‌ای، دارای نفی پیچیده، غلط تایپی یا احساس ترکیبی اشتباه کند.
- نتیجه تست نباید برای تنظیم threshold یا تغییر مدل استفاده شود.
- اجرای آینده روی ده درصد داده باید به‌عنوان نسخه جدید مدل منتشر شود.
