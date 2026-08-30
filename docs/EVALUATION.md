# Evaluation Report: Purchase Recommendation Status Classifier

## Executive decision

The released model is **`digikala-rec-xlm-roberta-10pct-v2.0.0`**, a fully fine-tuned `FacebookAI/xlm-roberta-base` classifier for three labels:

- `recommended`
- `not_recommended`
- `no_idea`

The release decision is **`PASS_PROMOTE_V2`**. On the locked test set, v2 reached **0.729173 Macro-F1**, compared with **0.717170** for v1 and **0.661141** for the selected classical baseline. The paired 95% bootstrap interval for the v2-minus-v1 Macro-F1 difference was **[0.003630, 0.020183]**, entirely above zero.

Macro-F1 is the primary metric because the test set is imbalanced: 78.78% of its rows are `recommended`. Accuracy alone would hide weak performance on the two minority classes; a majority-only baseline already obtains 77.24% validation accuracy while its validation Macro-F1 is only 0.290520.

This report covers the work required by project Section 3, purchase recommendation status prediction, and the evaluation work required by Section 4. It reports only evidence produced by the executed notebooks and published artifacts. It does not claim that model scores are calibrated probabilities, and it does not claim that v2 received a completed human review.

## System under evaluation

This component is a discriminative NLP classifier, not a conversational LLM. It is intended to run as a tool beside the final LLM system: it converts a review's title, body, advantages, and disadvantages into one of the three recommendation labels. An orchestrator can then aggregate these labels and provide structured evidence to the LLM.

The selected backbone is XLM-RoBERTa base:

- pinned checkpoint: `FacebookAI/xlm-roberta-base`
- pinned revision: `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`
- reported parameter count during candidate evaluation: 278,045,955
- 12 Transformer encoder layers, 768 hidden dimensions, and 12 attention heads
- a three-class sequence-classification head added for this task
- full fine-tuning; encoder parameters were not frozen
- class-weighted cross-entropy loss
- maximum input length: 128 tokens for the released XLM-RoBERTa models
- preprocessing contract: `fa_light_v1`
- training and inference runtime: Hugging Face Transformers with PyTorch, fp16 on a Tesla T4

The text pipeline normalizes Unicode with NFKC, maps common Arabic letter forms to Persian forms, removes byte-order marks, collapses whitespace, and joins non-empty fields in this fixed order:

```text
[TITLE] ... [BODY] ... [ADVANTAGES] ... [DISADVANTAGES] ...
```

## Data provenance and leakage control

The source was pinned to:

- Hugging Face repository: `RadeAI/Digikala_comments_products`
- source file: `digikala-comments.csv`
- source revision: `89c3133b169c8d3793db8834f56f32fee33d9db0`
- source size: 1,278,526,959 bytes
- source SHA-256: `c7a8aa3020334fde8ec24944576a03fe5785e6fe12cd01042f5836632ddf8297`

The 10% sampling pass scanned 6,156,289 rows. It found 5,261,863 rows with a supported label, removed 2,848 repeated comment IDs and 10 empty-text rows, and retained 5,259,005 unique-ID, non-empty rows as the denominator for the 10% train target.

### Meaning of a text group

`text_group_id` is not a topic cluster. It is a SHA-1 identifier for normalized duplicate text. The normalized body is used when present; otherwise the complete tagged text is used. Rows with the same normalized text therefore share one group even when their row IDs differ.

All rows from one text group are kept in only one split. This prevents a repeated review from appearing in training and again in validation or test. Both row-ID overlap and text-group overlap were checked pairwise across all splits.

For the 10% expansion, the original 2% validation and test IDs and groups were frozen. New training groups were selected deterministically with seeded SHA-256 ordering from a 20% candidate pool. Within that candidate pool, 5,161 groups covering 37,030 rows had conflicting labels and were excluded. These conflict counts describe the candidate pool, not the entire source dataset.

### Final split profile

| Split | Rows | Text groups | `recommended` | `not_recommended` | `no_idea` |
|---|---:|---:|---:|---:|---:|
| Train | 525,900 | 470,819 | 368,130 | 73,626 | 84,144 |
| Validation | 9,941 | 8,249 | 7,678 | 993 | 1,270 |
| Locked test | 9,662 | 8,215 | 7,612 | 980 | 1,070 |

The 525,900-row training split was intentionally minority-enriched to exactly 70% `recommended`, 14% `not_recommended`, and 16% `no_idea`. This is a training design choice, not a claim about the class distribution of the full source dataset. The original 85,694-row 2% training split was preserved inside it, and 440,206 rows were added: 300,024 `recommended`, 65,467 `not_recommended`, and 74,715 `no_idea`.

The validation-stage class weights were:

| Class | Weight |
|---|---:|
| `recommended` | 0.476190 |
| `not_recommended` | 2.380952 |
| `no_idea` | 2.083333 |

The released final model was refit on train plus validation, 535,841 labeled rows, for the epoch count selected on validation. The 9,662-row test split was excluded from training and model selection.

## Experimental protocol

The experiment used a validation-first promotion protocol:

1. Classical candidate models were compared on the original 2% validation split.
2. ParsBERT and XLM-RoBERTa encoder candidates were fully fine-tuned and compared on that same validation split.
3. XLM-RoBERTa was selected by validation Macro-F1, refit on train plus validation, and evaluated on the locked test to create v1.
4. The training split was expanded to 10% while validation and test remained unchanged.
5. The v2 release candidate was trained for two epochs and evaluated only on validation. Its best validation point was step 16,000, epoch 1.947032.
6. The validation gate returned `PASS_TO_FINAL`; only then was the final phase allowed.
7. A fresh final model was fit on the expanded train-plus-validation data for the selected epoch count.
8. Stored v1 predictions were aligned one-to-one to the unchanged test rows and groups, v1 metrics were reproduced, and v2 was evaluated once on the locked test.
9. The release decision combined metric gain, paired group-bootstrap uncertainty, latency, class-level advisory checks, and artifact integrity.

The validation and test evidence must not be mixed. Validation selected the model and epoch count. The locked test produced the final performance claim.

## Model selection results

### Validation comparison

| Stage | Candidate | Validation Macro-F1 | Weighted-F1 | Accuracy | Decision role |
|---|---|---:|---:|---:|---|
| Naive reference | Majority class | 0.290520 | 0.673155 | 0.772357 | Lower bound only |
| Classical | Body word TF-IDF + LinearSVC | 0.654200 | 0.820000 | 0.819133 | Rejected |
| Classical | Full-text char TF-IDF + LinearSVC | 0.664270 | 0.824158 | 0.820139 | Rejected |
| Classical | Full-text word+char TF-IDF + LinearSVC | 0.677072 | 0.833502 | 0.833115 | Rejected |
| Classical | Full-text word TF-IDF + LinearSVC | **0.680620** | 0.835309 | 0.834825 | Selected classical baseline |
| 2% encoder | ParsBERT | 0.721923 | 0.842941 | 0.827281 | Rejected |
| 2% encoder | XLM-RoBERTa base | **0.733375** | 0.849030 | 0.835027 | Promoted to v1 test |
| 10% encoder | XLM-RoBERTa base v2 candidate | **0.743702** | 0.854663 | 0.840760 | `PASS_TO_FINAL` |

The v2 candidate improved validation Macro-F1 over the v1 validation reference by **0.010326**, exceeding the required 0.005 gain. The validation split remained at 9,941 rows, and the test split was verified as untouched.

### Locked-test comparison

| Model | Macro-F1 | Weighted-F1 | Accuracy | Absolute Macro-F1 gain |
|---|---:|---:|---:|---:|
| Word TF-IDF + LinearSVC baseline | 0.661141 | 0.838370 | 0.841544 | reference |
| XLM-RoBERTa v1, 2% train | 0.717170 | 0.855353 | 0.842269 | +0.056029 vs baseline |
| XLM-RoBERTa v2, 10% train | **0.729173** | **0.860604** | **0.847030** | +0.068032 vs baseline; +0.012002 vs v1 |

The initial encoder-training notebook displayed v1 test Macro-F1 0.717487. The dedicated release evaluation reloaded the saved artifact and obtained 0.717170, a difference of 0.000316, within its predeclared 0.001 reproduction tolerance. The official v1 release metric is 0.717170. The v2 final notebook later reproduced that official value exactly from the published v1 predictions before comparing v2.

## Per-class locked-test results

| Class | Support | Baseline F1 | v1 F1 | v2 precision | v2 recall | v2 F1 |
|---|---:|---:|---:|---:|---:|---:|
| `recommended` | 7,612 | 0.923418 | 0.921837 | 0.964546 | 0.886364 | **0.923804** |
| `not_recommended` | 980 | 0.699237 | 0.759481 | 0.765244 | 0.768367 | **0.766802** |
| `no_idea` | 1,070 | 0.360769 | 0.470194 | 0.406417 | 0.639252 | **0.496912** |

The most important improvement is on the weakest class. Relative to v1, v2 raised `no_idea` recall from 0.589720 to 0.639252 and F1 from 0.470194 to 0.496912. The `recommended` and `not_recommended` F1 scores also improved slightly.

## Confusion matrices

Rows are observed labels and columns are model predictions, in this order: `recommended`, `not_recommended`, `no_idea`.

### v1 counts

| Observed / predicted | `recommended` | `not_recommended` | `no_idea` |
|---|---:|---:|---:|
| `recommended` | 6,746 | 64 | 802 |
| `not_recommended` | 38 | 761 | 181 |
| `no_idea` | 240 | 199 | 631 |

### v2 counts

| Observed / predicted | `recommended` | `not_recommended` | `no_idea` |
|---|---:|---:|---:|
| `recommended` | 6,747 | 60 | 805 |
| `not_recommended` | 33 | 753 | 194 |
| `no_idea` | 215 | 171 | 684 |

### v2 row-normalized interpretation

| Observed class | Correct | Predicted `recommended` | Predicted `not_recommended` | Predicted `no_idea` |
|---|---:|---:|---:|---:|
| `recommended` | 88.64% | 88.64% | 0.79% | 10.58% |
| `not_recommended` | 76.84% | 3.37% | 76.84% | 19.80% |
| `no_idea` | 63.93% | 20.09% | 15.98% | 63.93% |

v2 made 1,478 errors, down from 1,524 for v1. The largest remaining error channel is `recommended` to `no_idea` with 805 rows. The reverse ambiguity also remains substantial: 386 observed `no_idea` rows were assigned to one of the two decisive classes.

## Statistical uncertainty

Both release comparisons used 1,000 paired bootstrap iterations. The resampling unit was `text_group_id`, not an individual row. Every bootstrap draw therefore resampled duplicate-text groups as units and applied the same group multiplicities to both models. This respects within-group dependence and makes the model-difference interval paired.

| Comparison | Candidate Macro-F1 95% CI | Reference Macro-F1 95% CI | Paired delta 95% CI | Probability candidate was better |
|---|---|---|---|---:|
| v1 vs classical baseline | [0.706132, 0.729057] | [0.648623, 0.675333] | [0.042282, 0.069813] | 1.000 |
| v2 vs v1 | [0.716850, 0.740869] | [0.706132, 0.729057] | [0.003630, 0.020183] | 0.997 |

The v2 paired-delta interval is above zero but close to zero at its lower edge. The result supports promotion on this fixed test set; it should not be interpreted as a guarantee for future data distributions.

## Slice evaluation

The published v2 slice artifact evaluates 12 overlapping views of the same locked test set. Each row reports row count, distinct text-group count, Macro-F1, Weighted-F1, accuracy, and complete per-class precision, recall, F1, and support.

| Slice | Rows | Groups | v1 Macro-F1 | v2 Macro-F1 |
|---|---:|---:|---:|---:|
| All test rows | 9,662 | 8,215 | 0.717170 | **0.729173** |
| Title missing | 4,299 | 3,636 | 0.702890 | **0.717273** |
| Title present | 5,363 | 4,778 | 0.725610 | **0.735616** |
| Structured advantages or disadvantages present | 1,291 | 1,225 | 0.715052 | **0.724030** |
| Structured advantages and disadvantages missing | 8,371 | 7,065 | 0.717213 | **0.729710** |
| Truncated above 128 tokens | 87 | 87 | 0.674520 | **0.699779** |
| Not truncated | 9,575 | 8,128 | 0.717497 | **0.729397** |
| Very short, 1-16 tokens | 3,348 | 2,292 | 0.703551 | **0.719921** |
| Short, 17-48 tokens | 5,009 | 4,783 | 0.717336 | **0.727048** |
| Medium, 49-96 tokens | 1,070 | 1,067 | 0.728021 | **0.730025** |
| Long, 97-128 tokens | 148 | 148 | 0.682665 | **0.759871** |
| Above 128 tokens | 87 | 87 | 0.674520 | **0.699779** |

`truncated` and `length::truncated_gt_128` intentionally describe the same 87 rows through two naming schemes. The small long-text and truncated slices have high sampling uncertainty; their point estimates should not be treated as stable production guarantees. Title-missing reviews remain weaker than title-present reviews, and very short reviews remain weaker than the full test set.

## Error analysis and human review status

The v2 release contains `recommendation_10pct_failure_cases.csv`, covering all 1,478 incorrect test rows. It includes IDs, observed and predicted labels, the original text fields, model score, token length, truncation status, structural-field flags, length bucket, and confusion pair. This supports targeted qualitative analysis and future data collection.

The v1 release includes a generated 60-row manual-review sample. That artifact is a review worksheet/sample; this report does not assume that all rows were adjudicated by a human. The v2 pipeline did **not** create a manual-review sample or a completed human-review result. No v2 human-accuracy, annotator-agreement, or label-noise statistic is claimed.

## Latency and resource evaluation

The comparable v1 and v2 benchmarks ran on a Tesla T4 with PyTorch `2.10.0+cu128`, Transformers `4.57.6`, fp16, and maximum length 128. Each timing includes tokenization, CPU-to-GPU transfer, and the forward pass, with CUDA synchronization. The model was warmed up ten times. Single-request results use 200 sampled test rows; batch throughput uses 1,024 rows at batch size 32.

| Measure | v1 | v2 | Interpretation |
|---|---:|---:|---|
| Single-request mean | 10.681 ms | 13.036 ms | v2 is slower |
| Single-request p50 | 10.440 ms | 12.706 ms | v2 is slower |
| Single-request p95 | 11.476 ms | 15.335 ms | v2 is 1.336x v1 |
| Single-request p99 | 13.956 ms | 21.605 ms | v2 is slower |
| Batch throughput, size 32 | 861.64 rows/s | 769.76 rows/s | v2 retains 89.3% of v1 throughput |
| Peak allocated GPU memory | 1,414,517,248 bytes | 1,436,406,272 bytes | 1.5% increase |

v2 passed the hard p95 budget of 250 ms, the batch-throughput advisory threshold of at least 80% of v1, and the memory advisory threshold of at most 1.25x v1. It did **not** pass the advisory requirement that single-request p95 stay within 1.25x v1. This failed advisory did not block release because the absolute latency gate passed.

The earlier v1 evaluation also reported a 1,134,368,192-byte model artifact, no API requests, no API tokens, no API cost, and 2.771 candidate-plus-final GPU hours for that experiment. v2 final training took 22,472.75 seconds in its final session; the separate validation run took 24,591.21 seconds. These are observed Kaggle T4 measurements, not cloud-cost estimates or service-level guarantees.

## Release gates

### v1 release

The v1 evaluation decision was `PASS`. All reported gates passed:

- saved-model metric reproduction within 0.001
- at least 0.02 test Macro-F1 gain over the classical baseline
- improved `no_idea` F1
- class-level non-regression checks for `recommended` and `not_recommended`
- single-request p95 within budget
- paired bootstrap delta lower bound above zero

### v2 validation gate

The v2 validation decision was `PASS_TO_FINAL`. All mandatory checks passed:

- at least 0.005 validation Macro-F1 gain over v1
- validation rows remained locked
- test rows remained untouched

All three class-level validation advisory checks also passed.

### v2 final release gate

The final decision was `PASS_PROMOTE_V2`. All hard gates passed:

- exact v1 metric reproduction within 0.001; observed delta was 0.0
- at least 0.005 test Macro-F1 gain; observed gain was 0.012002
- paired Macro-F1 delta interval lower bound above zero; observed lower bound was 0.003630
- single-request p95 below 250 ms; observed p95 was 15.335 ms

The class-level non-regression advisories, runtime-comparability check, batch-throughput advisory, and memory advisory passed. The relative single-request p95 advisory failed, as documented above.

## Observed labels, predictions, and score semantics

The evaluation target `recommendation_status` is the observed dataset label. `prediction_10pct` and the per-class score columns are model outputs. They must remain separate in downstream systems.

The softmax values are **uncalibrated model scores**. They are useful for ranking and error inspection but are not validated probabilities of correctness. No calibration method, expected calibration error, Brier score, or selective-prediction threshold was evaluated.

When this classifier is integrated into the larger LLM system, an existing observed label should be preserved with provenance such as `source = observed`; inference should fill only missing labels with provenance such as `source = model_prediction` and the model version. The LLM should consume structured labels and aggregates, not invent replacement labels.

## Public evidence and release artifacts

The project publishes four complementary Kaggle datasets:

1. [Digikala recommendation delivery source](https://www.kaggle.com/datasets/maslri/digikala-recommendation-delivery-source) - a small source-delivery bundle for v1. It contains the four original notebooks, inference and preprocessing code, tests, requirements, example request, integration material, and selected v1 evaluation documentation. It does not contain the large trained-model weights and is not the v2 deployment release.
2. [Digikala Recommendation Status XLM-RoBERTa v1](https://www.kaggle.com/datasets/maslri/digikala-recommendation-status-xlm-roberta-v1) - the complete v1 release. It contains the v1 model and tokenizer, source-delivery files, split and training summaries, locked-test predictions, per-class metrics, confusion matrix, slice and failure analysis, latency, integration contracts, a generated manual-review worksheet, release card, and integrity manifest. v2 uses its published test predictions as the paired comparison reference.
3. [Digikala recommendation 10% validation pass](https://www.kaggle.com/datasets/maslri/digikala-recommendation-10pct-validation-pass) - a one-file gate artifact containing `recommendation_10pct_validation_summary.json`. It records the v2 release candidate's validation metrics, chosen epoch and step, sampling audit, class weights, locked-row checks, and `PASS_TO_FINAL` decision. It is evidence for model selection, not a deployable model package.
4. [Digikala Recommendation Status XLM-RoBERTa 10% v2](https://www.kaggle.com/datasets/maslri/digikala-recommendation-status-xlm-roberta-10-v2) - the current production release, approximately 1.177 GB with 24 files. It contains the final model and tokenizer, locked-test predictions and metrics, per-class results, confusion matrix, all failure rows, 12-slice results, paired bootstrap results, latency results, data and split metadata, integration contract, release card, and SHA-256 manifests.

For a judge reviewing evaluation evidence, the most direct v2 files are:

| File in the v2 dataset | Evidence supplied |
|---|---|
| `evaluation/recommendation_10pct_evaluation_summary.json` | Complete final decision, metrics, confusion counts, gates, runtime, sampling audit, hashes |
| `evaluation/recommendation_10pct_test_predictions.csv` | Row-level observed labels, v1/v2 predictions, and v2 scores for all 9,662 test rows |
| `evaluation/recommendation_10pct_per_class.csv` | Precision, recall, F1, and support for each class |
| `evaluation/recommendation_10pct_confusion_matrix.png` | Count and row-normalized confusion matrices |
| `evaluation/recommendation_10pct_slice_results.csv` | Metrics for all 12 predefined slices |
| `evaluation/recommendation_10pct_failure_cases.csv` | All 1,478 misclassified rows and diagnostic fields |
| `evaluation/recommendation_10pct_bootstrap_results.json` | Paired group-bootstrap intervals and probability of improvement |
| `evaluation/recommendation_10pct_latency_results.json` | T4 single-request, batch-throughput, and memory observations |
| `evaluation/recommendation_10pct_release_card.md` | Short human-readable promotion decision |
| `metadata/recommendation_10pct_split_manifest.csv` | IDs, product IDs, text-group IDs, labels, and split assignments for 545,503 selected rows |
| `metadata/recommendation_10pct_data_audit.json` | Compact sampling and leakage-control audit |
| `RELEASE_MANIFEST.sha256` | Integrity hashes for packaged files |

The split manifest is not the full 5.26-million-row source and does not include review text. The compact data-audit and integration-contract files are intentionally limited; the executed notebooks and this report provide the broader methodology.

## Artifact identity and reproducibility

| Artifact | SHA-256 or digest |
|---|---|
| Source CSV | `c7a8aa3020334fde8ec24944576a03fe5785e6fe12cd01042f5836632ddf8297` |
| Expanded split manifest canonical digest | `a9a2fdd44c937b6bc0c9e2042b018b72b9dbfdd839c1a23b65dd851cfca221d3` |
| Frozen validation digest | `de8552bf7d278b44c71a7d30c34c3aeb58c124e216640a6880aeb38ed52bf5bb` |
| Frozen test digest | `bd6f5e47bf368a77eaf9fd2bf8f44416d0e1878c59b477fc7794be316fdae9b5` |
| v1 model directory | `f343a3bcee07c68beaabece504a9efd1f200661e376f0b5235c75c7c9c394cf4` |
| Published v1 test predictions used by v2 | `15d840026c212aafae010bd08a3962458cf02703265ced6e861eaec42af1d0b2` |
| v2 model directory | `bc99ce62255893faf29fa46c27b3faa798b59fd09d4d31b5ab5ee280df8c570d` |
| v2 test predictions | `f20eabe60134c77c57c6020c36a5db64fdbeb3d952e603178ccc4345e4f7ea57` |

The v2 clean packaging notebook copied 24 approved files without running training or inference and produced a 1,177,466,280-byte release directory. `RELEASE_MANIFEST.sha256` checks the packaged files. The source manifest inside `metadata/` covers the original final-run artifacts and has a narrower scope than the release manifest.

## Limitations and appropriate interpretation

- The test set is group-safe for exact normalized duplicate text, but this does not prove the absence of near-duplicate paraphrases or product-level correlation across splits.
- The same fixed test set was used to report v1 and later to decide v2 promotion. v2 hyperparameters were selected on validation, but repeated release decisions can still create indirect test-set familiarity. A future version should add a new untouched temporal or external holdout.
- Results measure one pinned Digikala source revision. Performance under domain shift, newer products, slang changes, or different review platforms is unknown.
- `no_idea` remains the weakest class at 0.496912 F1, despite the improvement.
- Score calibration, abstention thresholds, robustness attacks, fairness across user groups, and temporal drift were not evaluated.
- Slice metrics overlap and are descriptive. The 87-row truncated slice and 148-row long slice are too small for strong independent claims.
- v2 has complete automated error artifacts but no completed human evaluation. Label quality and ambiguous cases still require human review.
- Latency is a Kaggle Tesla T4 benchmark, not an end-to-end API measurement. Serialization, queueing, network, orchestration, LLM time, and cold starts are excluded.
- The published v2 package is sufficient for inference and audit of the reported evaluation, but reproducing training also requires the pinned source dataset and the executed training notebooks.

## Final assessment

The evidence supports v2 as the best tested release for this project. It improves the primary metric on validation and on the locked test, improves every class-level F1 score relative to official v1, reduces total errors, and has a positive paired group-bootstrap difference. Its T4 latency remains far below the hard budget, although the relative single-request p95 regression must be visible to integrators. The release is suitable as the recommendation-status tool in the larger LLM system, with the documented limitations and provenance rules.
