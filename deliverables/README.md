# Delivery status

نسخه فعلی پروژه در recommendation_prediction_v1 قرار دارد و فایل recommendation_prediction_v1_source.zip برای اضافه‌کردن به Kaggle آماده است.

- source ZIP SHA-256: 245E492C47E3A3C0F57A10192B100B800A1D3833158C1C2D630285DA258A1178
- automated evaluation decision: PASS
- current training scope: frozen 2% sample
- runtime model artifact: موجود در Kaggle output و غایب از workspace محلی

برای ساخت تحویل اجرایی کامل، Notebook شماره 04 را در Kaggle با سه Input اجرا کنید: source ZIP، خروجی Transformer، و خروجی Evaluation. خروجی نهایی recommendation_prediction_v1_final.zip خواهد بود.

## آزمایش کنترل‌شدهٔ ۱۰٪

Notebook شماره 05 در `notebooks/05_kaggle_xlm_roberta_10pct.ipynb` برای گسترش train به حدود ۱۰٪ آماده است. نسخهٔ عمومی ۲٪ تا زمان PASS کامل این آزمایش، نسخهٔ رسمی باقی می‌ماند.

1. در Kaggle یک T4 و Internet را فعال کنید.
2. Output baseline شامل `sampled_split_manifest.csv` و Dataset عمومی v1 را Add Input کنید.
3. با `RUN_PHASE = 'validation'` اجرا و Output را Save Version کنید.
4. فقط اگر `decision` برابر `PASS_TO_FINAL` بود، Saved Version را Add Input و Notebook را با `RUN_PHASE = 'final'` اجرا کنید.
5. بلوک `COPY THIS VALIDATION SUMMARY` یا `COPY THIS FINAL SUMMARY` را برای بررسی برگردانید.

validation و test نسخهٔ ۲٪ در این آزمایش ثابت‌اند. مدل ۱۰٪ از `FacebookAI/xlm-roberta-base` پایه آموزش می‌بیند و ادامهٔ مدل ۲٪ نیست. هدف اصلی بیشینه‌کردن `Macro-F1` است؛ به همین دلیل داده‌های افزوده group-safe، گروه‌های جدیدِ دارای label conflict حذف و ترکیب train به‌صورت ملایم به نفع دو کلاس کم‌تعداد (`70/14/16`) غنی می‌شود. این نتیجه اثر مشترک دادهٔ بیشتر و سیاست پاک‌سازی/نمونه‌گیری جدید است، نه فقط اثر خالص حجم داده. checkpointها فقط برای resume هستند و نباید در Dataset انتشار نهایی قرار گیرند.
