# Acceptance Checklist

This checklist records the state of the historical v1 delivery package. Checked items were complete before packaging; unchecked items were intended to be completed by the Kaggle packaging run or by human reviewers.

- [x] The selected model and current release scope are documented.
- [x] Macro-F1, class-level metrics, bootstrap results, latency, and limitations are reported.
- [x] The JSON contract and `label_id` order are fixed.
- [x] Training preprocessing is implemented in the inference module.
- [x] A completely empty input is rejected.
- [x] The model does not overwrite an observed label.
- [x] Scores are explicitly identified as uncalibrated.
- [x] A Kaggle packaging notebook is provided for building the executable delivery.
- [ ] The Kaggle `best_transformer_encoder` directory has been inserted into the final package.
- [ ] All Kaggle training and evaluation artifacts have been inserted into the package.
- [ ] The runtime test has passed against the packaged artifact.
- [ ] The final `MANIFEST.sha256` has been generated.
- [ ] The final ZIP has been downloaded from Kaggle and extraction has been verified.
- [ ] Two-person human review of the manual-review worksheet has been completed. This completes the human-evaluation evidence but does not block the automated `PASS` decision.

The later public v1 packaging notebook completed the model/artifact copy, runtime smoke test, and manifest steps. The human-review checkbox remains open; the repository does not claim completed human adjudication.
