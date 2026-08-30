# Historical v1 Report: Project Sections 3 and 4

## Delivery scope

This report documents the historical v1 delivery for purchase recommendation status prediction. The task is supervised three-class classification of Digikala reviews. It uses a fine-tuned XLM-RoBERTa encoder. The project's LLM components are separate and consume this classifier's structured output.

The current repository also contains a later v2 release trained with the expanded 10% split. For the current model decision and all v2 evidence, use the [repository evaluation report](../../../docs/EVALUATION.md). The results below intentionally remain the v1 evidence recorded by this delivery package.

## Data and experiment

- Source: `RadeAI/Digikala_comments_products`
- Data revision: `89c3133b169c8d3793db8834f56f32fee33d9db0`
- Comments SHA-256: `c7a8aa3020334fde8ec24944576a03fe5785e6fe12cd01042f5836632ddf8297`
- v1 working sample: 105,297 rows, approximately 2% of valid source rows
- Group-safe splitting by `text_group_id` to prevent exact normalized duplicate text from crossing splits
- Fixed label order: `recommended`, `not_recommended`, `no_idea`
- Primary selection metric: Macro-F1
- Preprocessing: `fa_light_v1`
- Maximum input length: 128 tokens
- Selected model: `FacebookAI/xlm-roberta-base`

## Main result

On the 9,662-row locked test set, official v1 Macro-F1 was 0.7172. The classical baseline reached 0.6611, an absolute improvement of 0.0560. The 95% paired group-bootstrap interval for this difference was [0.0423, 0.0698], and the bootstrap probability that the Transformer was better was 1.0.

| Metric | Classical baseline | XLM-RoBERTa v1 |
|---|---:|---:|
| Macro-F1 | 0.6611 | 0.7172 |
| Weighted-F1 | 0.8384 | 0.8554 |
| Accuracy | 0.8415 | 0.8423 |
| `recommended` F1 | 0.9234 | 0.9218 |
| `not_recommended` F1 | 0.6992 | 0.7595 |
| `no_idea` F1 | 0.3608 | 0.4702 |

The v1 release gate returned `PASS`. Accuracy remained almost unchanged, while Macro-F1 and the two minority-class results improved. Macro-F1 is therefore more informative than accuracy for this imbalanced task.

## Error analysis

v1 made 1,524 errors. Its largest confusion was observed `recommended` to predicted `no_idea`, with 802 rows or 52.6% of all errors. Only 102 errors were direct exchanges between `recommended` and `not_recommended`. The dominant issue was therefore assigning a decisive review to the ambiguous `no_idea` class, rather than directly reversing positive and negative intent.

The manual-review worksheet contains 60 rows, with 10 sampled rows for each confusion direction. It is not sampled in proportion to the overall error distribution, so its contents cannot estimate dataset-wide error rates. The review fields were left open for human annotation, and this repository does not claim that two-person adjudication was completed.

### Repository redaction notice

To keep the judge-facing Git repository English-only while retaining reproducible numeric evidence, the repository copy of `recommendation_manual_review_sample.csv` replaces Persian free-text values in `title`, `body`, `advantages`, `disadvantages`, and `text_full` with `[REDACTED_PERSIAN_TEXT]`. IDs, text-group IDs, observed labels, predictions, scores, token and character lengths, structural flags, buckets, confusion pairs, and review-schema columns remain unchanged. The public Kaggle v1 artifact remains the source for the original review text.

## Runtime efficiency

- Measurement device: Tesla T4
- Single-request p95: 11.48 ms
- Batch throughput: approximately 862 reviews per second
- Peak allocated GPU memory: approximately 1.41 GB
- Model artifact size: approximately 1.13 GB
- API requests, tokens, and cost: zero

The benchmark covers the local tokenizer, device transfer, and model forward pass. It does not include network, queue, LLM, orchestration, or cold-start latency.

## Limitations of v1

- v1 was trained on the 2% working split; v2 later expanded training to 10%.
- `no_idea` remained the weakest and most semantically ambiguous class.
- Softmax scores were not calibrated and are not guaranteed probabilities.
- Complex negation, irony, mixed sentiment, typographical noise, and title-body conflict remain difficult.
- Only 87 test rows were truncated, so long-text conclusions are uncertain.
- The locked test must not be used for threshold tuning, label cleaning, or further model selection.
- The manual-review worksheet is not a completed human-evaluation result.

## Integration with the final system

The orchestrator supplies `title`, `body`, `advantages`, and `disadvantages`. The component returns one of the three labels, three uncalibrated scores, model version, preprocessing version, and provenance source.

If a valid `recommendation_status` already exists, preserve it with `source = observed`. Run inference only for a new or unlabeled review, and mark the result with `source = model_prediction`. The LLM may consume the structured label, but it must not present an uncalibrated score as certain probability.

Changing label order, preprocessing, or maximum length requires a new component version.

## Historical delivery decision

The v1 model and the automated Section 4 evidence passed their release gates. Its executable package required the saved `best_transformer_encoder` directory and complete Kaggle artifacts to be assembled by notebook 04. That packaging run later completed with a successful smoke test and integrity manifest.

For current deployment, use `digikala-rec-xlm-roberta-10pct-v2.0.0` from the public v2 Kaggle dataset. Keep this v1 package for rollback, provenance, and paired comparison.
