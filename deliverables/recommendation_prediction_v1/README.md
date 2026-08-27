# Digikala Recommendation Prediction v1

این پوشه بسته تحویل بخش سوم پروژه، «پیش‌بینی وضعیت پیشنهاد خرید»، و سهم این مؤلفه از بخش چهارم است.

## وضعیت انتشار

- مدل منتخب: FacebookAI/xlm-roberta-base fine-tuned
- دامنه داده فعلی: نمونه دو درصد، 105,297 ردیف
- تست قفل‌شده: 9,662 ردیف در 8,215 گروه متنی
- Macro-F1: 0.7172
- baseline Macro-F1: 0.6611
- تصمیم ارزیابی: PASS
- اجرای آینده روی 10 درصد جزو این نسخه نیست.

## ساختار بسته

- src/recommendation_prediction: ماژول inference آفلاین
- config/integration_contract.json: قرارداد اتصال با بخش‌های دیگر
- reports: گزارش بخش سوم و چهارم
- artifacts: summary فعلی و محل artifactهای Kaggle
- examples: ورودی نمونه
- tests: تست قرارداد preprocessing بدون نیاز به وزن مدل
- notebooks: سه Notebook آزمایش و Notebook نهایی بسته‌بندی در repository اصلی

پوشه مدل 1.13 گیگابایتی در workspace محلی موجود نیست. برای ساخت ZIP اجرایی، خروجی Notebookهای Transformer و Evaluation را به‌عنوان Kaggle Input به Notebook شماره 04 اضافه و آن را اجرا کنید. Notebook پوشه best_transformer_encoder و تمام شواهد ارزیابی را جمع می‌کند.

## اجرای inference پس از قرار دادن مدل

در ریشه همین بسته، وابستگی‌ها را نصب کنید:

    python -m pip install -r requirements.txt

سپس src را به PYTHONPATH اضافه کرده و نمونه را اجرا کنید:

    python -m recommendation_prediction.predictor \
      --model-dir model/best_transformer_encoder \
      --input-json examples/example_request.json

ورودی چهار فیلد متنی title، body، advantages و disadvantages دارد. در schema نسخه 1 همه این فیلدها string هستند. اگر هر چهار خالی باشند درخواست رد می‌شود.

## قرارداد کلاس‌ها

ترتیب کلاس‌ها بخشی از قرارداد مدل است و نباید تغییر کند:

1. recommended
2. not_recommended
3. no_idea

خروجی scores جمعی نزدیک به یک دارد، اما این مقادیر کالیبره نیستند و نباید به‌عنوان احتمال واقعی یا معیار تصمیم حساس استفاده شوند.

## اتصال به قسمت‌های LLM

این encoder یک مؤلفه مستقل NLP است. orchestrator نظر را به آن ارسال می‌کند و خروجی JSON را کنار خروجی بخش‌های LLM قرار می‌دهد. اگر recommendation_status واقعی وجود دارد، متد resolve آن را با source=observed حفظ می‌کند؛ در غیر این صورت پیش‌بینی با source=model_prediction تولید می‌شود.

## بازتولید

- seed: 42
- preprocessing: fa_light_v1
- max_length: 128
- dataset revision: 89c3133b169c8d3793db8834f56f32fee33d9db0
- dataset SHA-256: c7a8aa3020334fde8ec24944576a03fe5785e6fe12cd01042f5836632ddf8297
- base-model revision: e73636d4f797dec63c3081bb6ed5c7b0bb3f2089

revision مدل پایه fingerprint وزن fine-tuned نیست. بسته‌ساز نهایی SHA-256 فایل وزن fine-tuned را در MANIFEST ثبت می‌کند و predictor نیز آن را در خروجی برمی‌گرداند.

## محدودیت‌های رسمی

- نتیجه فقط مربوط به نمونه دو درصد است.
- no_idea ضعیف‌ترین کلاس باقی مانده است.
- تست برای tuning بعدی استفاده نمی‌شود.
- بازبینی 60 خطا stratified و غیرنماینده کل خطاهاست.
- confidence خام و کالیبره‌نشده است.

