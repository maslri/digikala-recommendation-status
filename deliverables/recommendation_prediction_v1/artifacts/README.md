# Artifact placement

وزن مدل و خروجی‌های کامل اجرا به علت حجم بالا در repository محلی نگهداری نشده‌اند. Notebook شماره 04 آن‌ها را مستقیماً از Kaggle Inputs پیدا می‌کند و بسته اجرایی نهایی می‌سازد.

موارد الزامی runtime:

- model/best_transformer_encoder/config.json
- model/best_transformer_encoder/model.safetensors یا pytorch_model.bin
- تمام فایل‌های tokenizer
- model/best_transformer_encoder/inference_config.json

موارد الزامی evidence:

- transformer_run_summary.json
- transformer_validation_results.csv
- sampled_split_manifest.csv
- recommendation_evaluation_summary.json
- recommendation_per_class.csv
- recommendation_slice_results.csv
- recommendation_confusion_matrix.png
- recommendation_failure_cases.csv
- recommendation_manual_review_sample.csv
- recommendation_latency_results.json
- recommendation_integration_contract.json
- recommendation_release_card.md

