# 09 - Fine-Tuning and Parameter-Efficient Fine-Tuning

## Goal

Decide whether weights should change, build valid supervised data, understand full fine-tuning/LoRA/QLoRA, estimate trainable parameters and memory, and evaluate gains plus regressions.

## Page-by-page lesson

### Page 1 - Persistent change

Fine-tuning changes parameters, so behavior can persist beyond the current prompt. This raises the evidence bar: define the gap, baseline simpler solutions, and create evaluation before training.

### Page 2 - Four questions

Ask whether weights should change, what objective/data should train, which update method fits resources, and how target gains and regressions will be measured.

### Page 3 - One lever among several

Prompting fixes specification, RAG supplies changing knowledge, and tools supply actions. Tune only when a stable repeated behavior/domain gap remains.

### Page 4 - Gradient updates persist

Training computes gradients of a loss and updates \(\theta\) or an adapter \(\Delta\theta\). The resulting checkpoint changes responses to many future inputs, including cases not present in training.

### Page 5 - Behavior versus knowledge

Current facts belong in RAG; exact calculations belong in tools; repeated format/style/task behavior may belong in SFT; domain language distribution may benefit from continued pretraining.

### Page 6 - Objective before method

Continued pretraining predicts tokens in unlabeled domain text. SFT predicts desired response tokens from instruction examples. Preference optimization compares or scores responses. LoRA is an update parameterization, not an objective.

### Page 7 - Continued pretraining (CPT)

CPT keeps the language-model objective but changes the text distribution. It can improve domain terminology and patterns, yet does not directly teach a helpful response format and may damage general capability if data is narrow.

### Page 8 - Instruction tuning

Multi-task instruction-response examples teach the model to infer tasks from natural-language instructions. Breadth and diversity can support transfer to unseen task forms.

### Page 9 - Domain fit versus assistant behavior

A model that predicts medical prose well may still answer users poorly. CPT changes distributional familiarity; SFT supplies demonstrations of desired interaction. Some projects use CPT followed by SFT.

### Page 10 - Dataset formats

Prompt/completion and structured chat records are converted into one token sequence. The exact serializer, role tokens, end tokens, truncation, and loss mask determine the actual training example.

### Page 11 - Chat templates

Training and inference must use the same role and turn conventions expected by the base checkpoint. Double-added special tokens or mismatched templates can cause major degradation.

### Page 12 - Loss masking

Completion-only training often masks system/user tokens and computes loss on assistant tokens. Full-sequence loss trains prediction of the prompt too. Masking is a recipe choice; verify eligible-token counts.

### Page 13 - Data quality

High-signal examples are correct, representative, diverse, consistent, and targeted to a measured gap. Large noisy datasets can teach contradictions, verbosity, artifacts, and unsafe shortcuts.

### Page 14 - Data-pipeline failures

Validate roles, template rendering, truncation, empty targets, duplicates, contamination, label consistency, train/validation leakage, language/domain mix, and a sample of decoded final sequences.

### Page 15 - Full fine-tuning memory

Every trainable weight needs gradients and optimizer state; activations remain for backward. Full tuning provides maximum flexibility but demands high memory, communication, and checkpoint storage.

### Page 16 - PEFT

PEFT freezes most base parameters and trains a small task-specific parameter set. It lowers optimizer/gradient memory and adapter storage, though forward activations and base weights still exist.

### Page 17 - LoRA idea

For a frozen matrix \(W_0\), LoRA learns a low-rank update \(BA\) and uses \(W'=W_0+sBA\). The base stays shared; small adapters store task-specific updates.

### Page 18 - Factorization

If \(W_0\in\mathbb{R}^{d_{out}\times d_{in}}\), then commonly \(A\in\mathbb{R}^{r\times d_{in}}\) and \(B\in\mathbb{R}^{d_{out}\times r}\). Trainable values are \(r(d_{in}+d_{out})\), much less than \(d_{in}d_{out}\) when \(r\) is small.

### Page 19 - Rank

Rank controls maximum update rank and parameter count. It is a capacity hyperparameter, not a percentage of learned knowledge. Higher rank can increase capacity, memory, overfitting risk, and optimization needs without monotonic quality gains.

### Page 20 - Alpha and learning rate

Many implementations scale by \(\alpha/r\) (variants exist). Alpha changes the contribution of the adapter output; learning rate changes update steps. Tune and report them together with dropout, initialization, and target modules.

### Page 21 - Target modules

Adapters may target query/value projections only or broader attention and MLP projections. Names are architecture-specific. Broader coverage increases capacity and state; inspect the model and verify trainable parameter names.

### Page 22 - Parameter-count example

For a 4,096×4,096 matrix and rank 16, full update has 16,777,216 parameters; LoRA has \(16(4096+4096)=131,072\), about 128 times fewer for that matrix.

### Page 23 - QLoRA

QLoRA stores the frozen base model in a 4-bit quantized representation and trains LoRA adapters, typically with higher-precision computation where needed. It attacks base-weight storage in addition to limiting trainable state.

### Page 24 - Gradient path through a frozen base

Base weights participate in forward/backward computations so gradients reach adapters, but they are not updated. Quantized storage, dequantization/computation dtype, adapter dtype, and optimizer state are distinct.

### Page 25 - Method comparison

Full FT maximizes trainable flexibility and storage cost. LoRA shares a normal-precision base and small adapters. QLoRA lowers base storage and enables smaller hardware, potentially with quantization/kernel trade-offs.

### Page 26 - Largest levers

Start with data correctness, template/mask, target modules, learning rate, effective batch, sequence length, and evaluation. Do not launch a large hyperparameter sweep on a broken pipeline.

### Page 27 - Batch and sequence accounting

Effective example batch is `microbatch × accumulation × data-parallel workers`; token count also depends on lengths and packing. Sequence length strongly affects activations, padding waste, and truncation.

### Page 28 - Hidden regressions

Target-task gains can coexist with lost general capability, style collapse, safety regressions, language regressions, hallucination, or memorized wording. Compare against the unchanged base on held-out suites.

### Page 29 - Respond to failure signals

Copied outputs suggest dedup/overfitting; validation divergence suggests fewer steps or better regularization/data; format errors suggest template/data; no improvement suggests wrong objective, data, modules, or insufficient signal—not automatically more rank.

### Page 30 - Domain and multilingual adaptation

Tokenizer efficiency, pretraining coverage, CPT data, and SFT examples all influence language results. Maintain general and language-specific validation and test code-switching where relevant.

### Page 31 - Layered evaluation

Measure target behavior, general capability, safety, calibration/grounding, multilingual behavior, robustness, latency/memory, and maintainability. Training loss only measures the optimized token objective.

### Page 32 - Choosing a lever

Inconsistent labels may justify SFT if prompts fail; stale policy facts still require RAG; calculation errors require tools; unstable JSON may require structured decoding before tuning.

### Page 33 - Measurability first

Fine-tuning creates data, checkpoint, serving, monitoring, rollback, and refresh obligations. If the gap has no reliable metric or examples, a run cannot establish success.

### Page 34 - Four-question synthesis

Should weights change? What objective and supervised tokens? Which trainable state and hardware path? Which target and regression evaluations? Answer all four in an experiment card.

### Page 35 - Sources

Use primary papers for LoRA/QLoRA/objectives and current library docs for exact APIs, target names, quantization support, and template behavior.

## Worked example - LoRA parameter count

For `q_proj` with \(d_{in}=d_{out}=4096\), rank 8:

\[
8(4096+4096)=65{,}536
\]

trainable parameters versus 16,777,216 full parameters. Across 32 layers, adapting one such matrix gives 2,097,152 adapter parameters, excluding biases and other targets.

## Worked example - Effective batch

Microbatch 2 × gradient accumulation 16 × 4 GPUs = 128 examples/update. If average non-padding length is 700 tokens, that is about 89,600 useful tokens/update. Report both examples and tokens.

## Minimal experiment card

```text
Measured gap: intent macro-F1 0.74; target >=0.84
Simpler baselines: prompt 0.78; RAG not applicable
Base/template: exact model ID and chat template version
Data: 8k train / 1k validation, deduplicated by customer thread
Method: LoRA; rank/alpha/dropout/targets declared
Controls: LR, scheduler, effective tokens/update, max length, seed
Release gates: target F1, general suite, Persian suite, safety, latency
Rollback: retain base and previous adapter
```

## Practice

1. Choose CPT, SFT, RAG, tool, or no change for five diagnosed gaps.
2. Render and manually inspect ten training examples after tokenization/masking.
3. Calculate LoRA parameters for a 4096×11008 MLP matrix at rank 16.
4. Design target and regression evaluations for a legal-document classifier.
5. Explain QLoRA without saying it “trains the 4-bit weights.”

## Mastery check

You are ready when you can defend *not* tuning, calculate adapter state, validate the actual tokenized objective, and define rollback-worthy regression gates.

