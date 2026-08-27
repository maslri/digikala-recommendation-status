# 05 - Scaling Laws and Model Families

## Goal

Reason about parameters, data, compute, training/inference cost, dense versus mixture-of-experts models, openness, and multilingual evaluation.

## Page-by-page lesson

### Page 1 - Central decision

With a fixed budget, you must divide resources among model capacity, training data, and compute. Parameter count alone cannot answer whether a run is well designed.

### Page 2 - Five questions

The session moves from empirical scaling relationships to compute-optimal training, deployment-aware choices, model-family differences, and selection strategy.

### Page 3 - What to buy

A larger undertrained model may lose to a smaller model trained on more data. State the constraint first: training FLOPs, available tokens, wall time, memory, serving volume, latency, or quality target.

### Page 4 - N, D, and C

\(N\) is model parameters, \(D\) training tokens, and \(C\) training compute. For dense autoregressive Transformers, \(C\) is roughly proportional to \(ND\), with architecture- and implementation-dependent constants.

### Page 5 - Smooth scaling

Empirical studies found loss often improves predictably as parameters, data, or compute increase over broad ranges. “Smooth” describes average pretraining loss; individual capabilities may appear noisy or threshold-like.

### Page 6 - Power laws

A simplified fit is \(L(x)=L_\infty + ax^{-b}\). The irreducible term \(L_\infty\) is a floor, and exponent \(b\) controls diminishing returns. Fits apply only near the observed regime and data distribution.

### Page 7 - Forecasting from small runs

Train a ladder of smaller models, fit loss versus compute, and forecast a larger run. This can guide budgets and detect runs that underperform expectation, but extrapolation uncertainty must be reported.

### Page 8 - Compute-optimal point

For each compute budget, many \((N,D)\) combinations are possible. IsoFLOP experiments find the combination with minimum loss. Too-large models see too few tokens; too-small models lack capacity.

### Page 9 - Chinchilla lesson

Chinchilla showed a smaller model trained on substantially more data could outperform a much larger, undertrained model at similar compute. The lesson is balance, not “small always wins.”

### Page 10 - Tokens-per-parameter heuristic

The famous roughly 20 tokens per parameter is tied to a particular study and objective. Data quality, repeated epochs, architecture, optimizer, inference costs, and modern recipes can shift the optimum.

### Page 11 - Serving-aware optimum

Training happens once; inference may happen billions of times. A smaller model trained longer can cost more initially but save memory and latency on every request. Optimize lifecycle cost, not only pretraining loss.

### Page 12 - Limits of scaling laws

Loss cannot guarantee factuality, safety, tool reliability, a language-specific score, or product usefulness. Scaling laws forecast selected metrics under assumptions; product evaluations remain necessary.

### Page 13 - Model families

A family may include several sizes, base and instruction checkpoints, context lengths, quantizations, modalities, licenses, and deployment targets. Compare exact variants rather than family names.

### Page 14 - Dense versus MoE

Dense models apply the same main weights to every token. Mixture-of-experts layers route each token to a small subset of experts, increasing stored capacity without activating all expert parameters per token.

### Page 15 - Total and active parameters

Total parameters affect storage and distributed communication. Active parameters affect per-token arithmetic more directly. Both counts matter; comparing a dense model's total parameters with an MoE's total alone can mislead.

### Page 16 - MoE systems costs

Routing can create load imbalance, expert communication, memory-placement challenges, and capacity overflows. Sparse arithmetic does not guarantee low latency on every hardware/software stack.

### Page 17 - Comparison dimensions

Evaluate architecture, quality by task/language, context behavior, memory, throughput, latency, tool support, license, weights, training transparency, ecosystem, and operational constraints.

### Page 18 - Open-weight versus open-source AI

Accessible weights allow local inference and adaptation, but reproducibility may also require data, training code, recipe, and rights. “Open” should be decomposed into specific freedoms and disclosed components.

### Page 19 - Multilingual claims

A supported-language count says little about token efficiency or quality. Test native prompts, dialects, domains, code-switching, safety, and cultural context with language-specific evaluation sets.

### Page 20 - Strategy exercise

Choose a portfolio: perhaps a small local model for routing, a stronger model for hard requests, and retrieval for changing knowledge. State fallback and evaluation criteria.

### Page 21 - Synthesis

Scale improves average prediction; balance N/D/C; serving changes the optimum; model families hide multiple dimensions; sparse models trade arithmetic for systems complexity; evaluation must match deployment.

### Page 22 - Sources

Use scaling papers for fitted relationships and official model cards for exact architecture/license facts. Check dates and model versions before deployment decisions.

## Worked example - Two budgets

Assume the crude compute proxy \(C=6ND\).

- Plan A: \(N=10\)B, \(D=100\)B → proxy \(6{,}000\)B² operations.
- Plan B: \(N=5\)B, \(D=200\)B → the same proxy compute.

These plans can have different loss, memory, and serving cost. Scaling experiments choose between them; the formula alone does not.

## Worked example - Lifecycle cost

Model A costs $1M to train and $0.002/request. Model B costs $1.4M to train and $0.001/request. Break-even request count is:

\[
(1.4M-1.0M)/(0.002-0.001)=400M\text{ requests}.
\]

Below that volume A is cheaper; above it B is cheaper, before other costs.

## Practice

1. Explain diminishing returns from a power law.
2. Compare a 30B dense model with a 100B-total/15B-active MoE without declaring a winner from counts alone.
3. Create a multilingual evaluation plan for Persian and English customer support.
4. List the exact information needed before calling a checkpoint “open.”

## Mastery check

You are ready when you can reject parameter-count-only comparisons and defend a model strategy using training, serving, quality, and governance constraints.

