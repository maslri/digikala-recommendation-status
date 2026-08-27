# 04 - The Pretraining Pipeline

## Goal

Trace raw documents into a reusable base-model checkpoint, including governance, curation, optimization, and evaluation risks.

## Page-by-page lesson

### Page 1 - Pipeline overview

Pretraining is a chain: collect → extract → filter → deduplicate → tokenize/pack → train → checkpoint/evaluate. Upstream data choices define the signal long before GPU training begins.

### Page 2 - Roadmap

Separate four questions: what objective is optimized, what data supplies examples, how the data is transformed, and how the optimization run is controlled. This separation makes failures diagnosable.

### Page 3 - Base model is not a chatbot

Pretraining teaches continuation over a broad distribution. Instruction following, conversational roles, refusals, and preference behavior usually require later supervised and preference-based post-training.

### Page 4 - Data and learning pipelines

The data pipeline creates an intentional token distribution. The learning loop repeatedly turns prediction error into parameter updates. A perfect optimizer cannot recover information removed or corrupted upstream.

### Page 5 - Data sources

Web, books, code, encyclopedias, papers, and domain corpora contribute different structures and risks. Licensing, privacy, provenance, language balance, and temporal coverage are design constraints, not afterthoughts.

### Page 6 - Mixture design

Raw availability is not the training mixture. Sampling weights upsample scarce but valuable sources and downsample dominant ones. Because every sampled token contributes gradient signal, mixture weights shape learned behavior.

### Page 7 - Extracting web text

HTML contains menus, ads, scripts, cookie banners, and repeated templates. Main-content extraction should retain useful document structure while removing boilerplate. Parser mistakes can silently delete tables, headings, or code.

### Page 8 - Cleaning and language ID

Rules and classifiers detect corruption, spam, unwanted content, personally identifiable information, and language. Filters have false positives and can disproportionately remove dialects or minority-language text, so their effects must be audited.

### Page 9 - Quality

“Quality” is a proxy defined by the training goal. Length, repetition, link density, fluency, and classifier scores can rank documents, but a filter trained on one notion of good writing may erase useful diversity.

### Page 10 - FineWeb case study

Open pipelines demonstrate that extraction, deduplication, and filtering recipes materially affect model quality. Treat dataset versions and processing code as part of the experiment configuration.

### Page 11 - Duplication

Mirrors, reposts, templates, and copied passages overweight repeated content. Repetition wastes token budget and raises memorization and benchmark-contamination risk.

### Page 12 - Exact and fuzzy deduplication

Hashes find byte-identical or normalized-identical documents. Shingles, MinHash, and locality-sensitive hashing approximate near-duplicate similarity at scale. Passage-level methods catch copied sections that document-level matching may miss.

### Page 13 - Why dedup matters

Training frequency acts like weight. Removing redundant copies increases unique information per token and improves the credibility of evaluation, but over-aggressive dedup can remove legitimate repeated forms.

### Page 14 - Benchmark contamination

If evaluation questions or solutions occur in training, scores may measure recall rather than generalization. Leakage can arrive through repositories, discussions, mirrors, papers, and dataset descriptions.

### Page 15 - Limits of decontamination

Exact matching misses paraphrases; fuzzy matching can flag innocent overlap; hidden test content cannot be searched. Report the method and uncertainty rather than claiming perfect cleanliness.

### Page 16 - Packing token blocks

Documents become token streams with boundary markers and are packed into fixed-length sequences. Packing reduces padding waste. Boundary and masking choices determine whether one document can attend to another and which tokens contribute loss.

### Page 17 - Teacher forcing

The known sequence supplies every previous target token during training. Input `[The, cat, sat]` creates targets `[cat, sat, ...]`. This is efficient but differs from inference, where the model consumes its own prior choices.

### Page 18 - Cross-entropy

For target token \(y\), loss is \(-\log p(y)\), averaged over eligible tokens and batches. Assigning 0.9 to the target gives low loss; assigning 0.01 gives high loss. Perplexity is often \(e^{\text{average loss}}\).

### Page 19 - Training loop

A step performs batch loading, forward pass, loss computation, backpropagation, optional gradient clipping, optimizer update, and scheduler update. Logging and validation must distinguish data, numerical, and optimization failures.

### Page 20 - Optimizer

Gradients estimate a local loss-reducing direction. Adam-like optimizers track moments; the learning-rate schedule controls update scale over time. Warmup helps early stability; decay reduces updates later.

### Page 21 - Accounting units

Parameters describe model size; training tokens describe data exposure; tokens per batch describe work per update; steps equal total processed tokens divided by global tokens per step. Keep sequences, tokens, examples, and updates distinct.

### Page 22 - Compute cost

A common dense-Transformer training estimate is proportional to \(N\times D\), often approximated near \(6ND\) FLOPs for parameter count \(N\) and training tokens \(D\). It is a planning approximation, not an exact invoice.

### Page 23 - Checkpoints

Resumable state can include weights, optimizer moments, scheduler, random-number generators, data position, scaler state, and metadata. A file that loads weights may still fail to reproduce the interrupted trajectory.

### Page 24 - Pipeline diagnosis

Memorized evaluation answers suggest contamination or duplication; broken code may suggest extraction/filtering; language weakness may suggest mixture/tokenization; unstable loss may suggest data corruption, precision, or optimization.

### Page 25 - Checklist

Record rights and provenance, privacy handling, source mixture, filter statistics, dedup method, tokenizer, packing, optimization configuration, evaluation sets, checkpoint/restore tests, and known limitations.

### Page 26 - Takeaways

A base model is the result of coordinated data, objective, optimization, and systems decisions. Model weights are compressed consequences of that entire pipeline.

### Page 27 - Sources

Reproduce claims using data cards, pipeline code, training reports, and primary papers. Dataset names without versions and recipes are insufficient for comparison.

## Worked example 1 - Cross-entropy and perplexity

For three targets with probabilities `[0.5, 0.25, 0.125]`:

\[
L=(-\ln .5-\ln .25-\ln .125)/3\approx1.386.
\]

Perplexity is \(e^{1.386}\approx4\). Informally, the model is as uncertain as choosing among four equally likely options, though real vocabulary distributions are not uniform.

## Worked example 2 - Steps

Suppose total data is 100 billion tokens. With global batch 2 million tokens per optimizer step:

\[
100{,}000{,}000{,}000 / 2{,}000{,}000 = 50{,}000\text{ steps}.
\]

If gradient accumulation uses 8 microsteps, there are 400,000 forward/backward microsteps but 50,000 optimizer updates.

## Worked example 3 - Deduplication

Normalize lowercase/whitespace, hash exact documents, then represent each document by 5-word shingles. MinHash can cheaply find candidate pairs; exact Jaccard similarity verifies them. Set thresholds using manual audits because copied templates and legitimate repeated legal text behave differently.

## Practice

1. Design a corpus mixture for a bilingual coding assistant and defend each source weight.
2. Compute loss for a target probability of 0.01 and compare it with 0.8.
3. List checkpoint state needed for deterministic-enough resume.
4. Diagnose: excellent benchmark score, poor paraphrases, and benchmark solutions found online.
5. Draft a minimal data card with provenance, filters, deduplication, language distribution, and limitations.

## Mastery check

You are ready when you can trace one web page into training blocks, calculate loss and steps, and identify where contamination, memorization, or instability could enter.

