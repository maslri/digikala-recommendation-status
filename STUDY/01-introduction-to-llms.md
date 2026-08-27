# 01 - Introduction to Large Language Models

## What you will learn

You will understand language modeling, autoregressive generation, decoding, model stages, strengths, and limitations. No machine-learning background is assumed.

## Page-by-page lesson

### Page 1 - Course topic

An LLM is a neural network trained on enormous collections of token sequences. “Large” usually refers to the number of learned parameters, the training data, and the compute used. “Language model” means it assigns probabilities to text sequences, usually by predicting the next token.

### Page 2 - Why LLMs matter

Older NLP systems often used a different pipeline and labeled dataset for each task. An LLM exposes a general language interface: the task, data, and desired format can be described in text. This does not mean one model is equally good at every task; it means one model can be *conditioned* to attempt many tasks.

### Page 3 - Classical NLP

A classical sentiment system might clean text, count words, create features such as TF-IDF, and feed them to a classifier. This can be cheaper, faster, and more predictable than an LLM for a narrow task. LLMs reduce manual feature engineering but add cost, nondeterminism, and grounding concerns.

### Page 4 - What is a language model?

Given tokens \(x_1,\ldots,x_t\), a causal language model estimates \(P(x_{t+1}\mid x_1,\ldots,x_t)\). The chain rule turns a whole sequence into a product of next-token probabilities:

\[
P(x_1,\ldots,x_n)=\prod_{t=1}^{n}P(x_t\mid x_{<t}).
\]

Translation or summarization can be represented as text continuation: put the instruction and input in the prefix, then predict the desired output.

### Page 5 - From n-grams to Transformers

An n-gram counts short local patterns; it cannot easily use distant context. Neural models learn continuous representations. RNNs process sequence state recurrently, but long paths and limited parallelism make scaling difficult. Transformers use attention, allowing direct token-to-token interactions and efficient parallel training.

### Page 6 - Autoregressive generation

Generation is a loop: predict a probability distribution, choose one token, append it, and repeat. Training can process all known target positions in parallel; inference cannot know future generated tokens, so decoding remains sequential.

### Page 7 - Sampling and decoding

The model supplies probabilities; a decoding algorithm makes the choice. Greedy decoding always chooses the maximum. Temperature rescales logits: low temperature sharpens the distribution; high temperature flattens it. Top-k keeps only the k most likely tokens. Top-p keeps the smallest set whose cumulative probability reaches p.

### Page 8 - Base, instruction, and chat models

A base model continues text. Supervised instruction tuning teaches examples of request-response behavior. Preference or safety post-training further shapes which responses are favored. All remain next-token predictors, but their learned conditional behavior differs.

### Page 9 - Why scale changed capability

More parameters can represent more patterns; more diverse tokens supply more evidence; more compute enables optimization. These resources must be balanced. Lower average prediction loss often improves many capabilities, but does not guarantee truthfulness, safety, or performance on a particular product task.

### Page 10 - Strengths

LLMs are strong at transformations expressed in language: drafting, rewriting, summarizing, translating, extracting, classifying, explaining code, and generating candidate solutions. Their best use often combines flexible interpretation with external validation.

### Page 11 - Weaknesses

Fluency is not evidence. Hallucination occurs because the objective rewards likely continuations, not a database-style truth lookup. Knowledge can be stale; prompts can be sensitive; long contexts can be used unevenly; memorized private or copyrighted sequences may create risk. High-stakes outputs require grounding and checks.

### Page 12 - Mental model

Remember three facts: the model predicts tokens, its output depends on available context, and plausible language can be wrong. This simple model explains why prompting, retrieval, tools, and evaluation matter later.

### Page 13 - References

References are not decoration: use primary papers for mechanisms and official documentation for current interfaces. Distinguish empirical findings, engineering guidance, and the instructor's synthesis.

## Worked example 1 - A tiny next-token model

Suppose after “I drink hot” the model assigns:

| Token | Probability |
|---|---:|
| tea | 0.55 |
| coffee | 0.30 |
| water | 0.10 |
| yesterday | 0.05 |

Greedy decoding returns `tea`. Sampling may return `coffee` or even `yesterday`. The probabilities are conditioned on the prefix; changing it to “Every morning I drink hot” may change every value.

If the correct training target is `tea`, its token loss is \(-\ln(0.55)\approx0.60\). Raising the assigned probability lowers the loss.

## Worked example 2 - Choosing the right tool

Task: label 50 million short transactions as `debit` or `credit` using a fixed rule. A small deterministic program is preferable. Task: interpret varied customer explanations and draft a response using policy passages. An LLM plus retrieval and validation is more suitable.

## Common misconceptions

- “The model searches its training data.” It usually generates from learned parameters; it does not retrieve a source record unless a retrieval system is added.
- “Temperature changes knowledge.” It changes selection from the output distribution, not the weights.
- “A larger model is automatically truthful.” Scale can improve prediction without guaranteeing grounded claims.
- “Chat models reason exactly like humans.” Human-like text is not proof of a human-like internal process.

## Practice

1. Explain next-token prediction to a 12-year-old without using “AI.”
2. Give one task where classical NLP is preferable and justify cost, accuracy, and reliability.
3. For probabilities `[0.6, 0.25, 0.1, 0.05]`, state the greedy choice and the top-p set for `p=0.8`.
4. Label each claim as strength, limitation, or misconception: “can summarize,” “always cites facts,” “depends on context,” “is a database.”

## Mastery check

You are ready when you can trace prompt → token probabilities → decoder choice → appended token, and explain why a fluent continuation may still be false.

