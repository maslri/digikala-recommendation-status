# Digikala Recommendation Status Prediction

این مخزن مربوط به بخش سوم پروژهٔ هوش مصنوعی کوئرا است: **پیش‌بینی وضعیت پیشنهاد خرید** برای دیدگاه‌های فارسی دیجی‌کالا.

مدل یک طبقه‌بند سه‌کلاسه است:

- `recommended`
- `not_recommended`
- `no_idea`

## وضعیت فعلی

- مدل رسمی v1: `XLM-RoBERTa-base` آموزش‌دیده روی نمونهٔ قفل‌شدهٔ ۲٪
- Test Macro-F1 رسمی v1: `0.71717`
- آزمایش ۱۰٪: validation را با Macro-F1 برابر `0.74370` پاس کرده است
- فاز final/test مدل ۱۰٪ هنوز باید یک بار اجرا شود

## Notebookها

1. `01_kaggle_classical_baselines.ipynb`: baselineهای کلاسیک
2. `02_kaggle_transformer_encoders.ipynb`: مقایسهٔ encoderهای Transformer
3. `03_kaggle_recommendation_evaluation.ipynb`: ارزیابی بخش چهارم
4. `04_kaggle_build_final_delivery.ipynb`: ساخت بستهٔ تحویل v1
5. `05_kaggle_xlm_roberta_10pct.ipynb`: آموزش و ارزیابی دو‌مرحله‌ای مدل ۱۰٪

## ساختار

- `notebooks/`: Notebookهای قابل‌اجرا در Kaggle
- `deliverables/`: کد inference، تست‌ها، قرارداد اتصال و گزارش‌ها
- `STUDY/`: یادداشت‌های دانشی مرتبط با پروژه
- `QBC12 _ AI _ Project 3.pdf`: صورت پروژه

## داده و مدل

دیتاست و وزن مدل به‌دلیل حجم زیاد داخل Git نگهداری نمی‌شوند. منبع داده pin شده است:

- Hugging Face: `RadeAI/Digikala_comments_products`
- Dataset مدل عمومی v1 در Kaggle: https://www.kaggle.com/datasets/maslri/digikala-recommendation-status-xlm-roberta-v1/data

جزئیات اجرای مدل و نحوهٔ اتصال آن به سیستم نهایی در `deliverables/help.md` قرار دارد.

