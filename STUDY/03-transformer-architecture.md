# 03 - Transformer Architecture

## Goal

Understand how token vectors become contextual vectors and next-token logits. You should be comfortable with vectors, matrix shapes, dot products, softmax, and weighted averages after this guide.

## Page-by-page lesson

### Page 1 - End-to-end architecture

The decoder-only path is token IDs → embeddings/positions → repeated Transformer blocks → vocabulary logits → decoder choice. Each block refines every token representation; the final state at the newest position is used to predict the next token.

### Page 2 - Attention versus recurrence

An RNN moves information through sequential hidden states. Self-attention gives each position a direct weighted connection to allowed positions, shortening information paths and enabling parallel training across known tokens.

### Page 3 - Contextual vectors

The embedding row for `bank` begins the same in “river bank” and “central bank.” After attention and MLP layers mix context, the states differ. Meaning is therefore represented by context-dependent hidden states, not by the initial embedding alone.

### Page 4 - Self-attention intuition

For every query position, attention computes how strongly it should use each allowed source position. The output is a weighted sum of source value vectors. Weights are learned dynamically from the current sequence.

### Page 5 - Query, key, and value

From input states \(X\), learned projections create \(Q=XW_Q\), \(K=XW_K\), and \(V=XW_V\). A query describes what a position seeks, a key describes how a source can be matched, and a value contains information to transfer. These are useful analogies, not human-readable database fields.

### Page 6 - Scaled dot-product attention

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V.
\]

Dot products create compatibility scores. Division by \(\sqrt{d_k}\) controls their scale. The mask \(M\) makes forbidden scores extremely negative. Row-wise softmax produces nonnegative weights summing to one.

### Page 7 - Attention matrix

For \(n\) tokens, \(QK^T\) is \(n\times n\). Row \(i\) describes where query token \(i\) reads. Column \(j\) corresponds to source token \(j\). Attention visualizations are useful clues but do not by themselves explain all computation.

### Page 8 - Multi-head attention

Each head has separate projections and operates in a smaller subspace. Head outputs are concatenated and projected with \(W_O\). Multiple heads give parallel channels for different matching patterns, though a head does not necessarily have one stable linguistic job.

### Page 9 - Benefits of heads

Heads can specialize in relative positions, syntax-like relations, entity associations, delimiters, or other learned patterns. More heads are not automatically better; hidden size, head dimension, grouped-query design, and hardware all interact.

### Page 10 - Causal masking

At position \(i\), a causal decoder may attend only to positions \(j\le i\). Future positions get zero probability after masking and softmax. This prevents information leakage during training while allowing all position computations to run in parallel.

### Page 11 - Original encoder-decoder Transformer

The original architecture has a bidirectional encoder, a causal decoder, and decoder cross-attention to encoder states. It is well matched to sequence-to-sequence tasks such as translation. Modern chat LLMs are often decoder-only.

### Page 12 - Parallel training, sequential generation

Training has the whole ground-truth sequence and computes losses for many positions simultaneously under the mask. Generation must produce token \(t\) before token \(t+1\), creating an inherently sequential outer loop.

### Page 13 - Modern decoder block

A common pre-normalized block performs `x = x + attention(norm(x))`, then `x = x + mlp(norm(x))`. Attention mixes positions; the MLP transforms features independently at each position; residual paths preserve and update state.

### Page 14 - Llama architecture example

Llama-style models combine decoder-only causal attention, RMSNorm, RoPE, gated MLPs such as SwiGLU, and efficient key/value-head variants. Exact dimensions and choices differ by model generation.

### Page 15 - Evolution from the original Transformer

The essential pattern stayed, but normalization order, activations, positional methods, attention heads, and implementation kernels evolved. “Transformer” names a family, not one immutable block.

### Page 16 - Feed-forward network

The MLP applies the same learned function to each position independently. A simple form is \(\operatorname{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2\). It usually expands the hidden dimension, applies a nonlinearity, and contracts it.

### Page 17 - Gated MLPs

SwiGLU-like layers create two projections; one passes through an activation and gates the other. Gating lets the network control feature flow and has performed well in large models.

### Page 18 - Residual connections

\(x_{out}=x+F(x)\) lets a sublayer learn an update instead of rebuilding the representation. The identity path supports gradient flow and deep optimization. Shape equality is required at the addition.

### Page 19 - Layer normalization

LayerNorm normalizes features within a token state. Post-LN applies normalization after residual addition; pre-LN normalizes before the sublayer. Pre-LN commonly improves optimization stability in deep models. RMSNorm uses root-mean-square scaling without mean subtraction.

### Page 20 - Position information

Self-attention without positional information is permutation-equivariant. Position mechanisms break that symmetry, allowing order and distance to influence scores. Context length behavior depends partly on how position was represented during training.

### Page 21 - RoPE

RoPE rotates pairs of query/key features by position-dependent angles. Dot products then carry relative-position information. It does not simply append a position number; it changes the geometry used by attention.

### Page 22 - Architecture families

Encoder-only models use bidirectional context and excel at representation tasks. Decoder-only models use causal attention and generation. Encoder-decoder models encode an input and autoregressively generate an output while cross-attending to it.

### Page 23 - Why decoder-only dominates generative LLMs

One causal next-token objective trains on arbitrary text and code. Instructions, examples, and retrieved documents can all be placed in a prefix, simplifying one general interface and scaling recipe.

### Page 24 - Context length complexity

Dense attention creates \(n^2\) score relationships and roughly \(O(n^2d)\) arithmetic. Doubling \(n\) makes four times as many pair scores. Other costs, such as MLP work and KV-cache storage, scale differently.

### Page 25 - KV cache

During decoding, past keys and values do not change. Caching them avoids recomputing the entire prefix at each step. Cache memory grows with layers, sequence length, batch/concurrency, key/value heads, head dimension, and bytes per value.

### Page 26 - Efficient attention concepts

FlashAttention computes exact dense attention in tiles to reduce expensive memory traffic. Multi-query attention shares one K/V head across query heads; grouped-query attention uses several K/V groups. These reduce cache and decode bandwidth with quality trade-offs.

### Page 27 - Full model path

Embeddings identify tokens, position signals encode order, attention mixes prior context, MLPs transform features, residuals preserve state, and final logits score vocabulary tokens. Repeating this block yields increasingly useful representations.

### Page 28 - Sources

Use the original Transformer paper for canonical equations, architecture reports for model-specific changes, and systems papers for kernel behavior. A high-level diagram cannot establish exact implementation details.

## Worked example 1 - One attention row

Let a query have scores `[2, 1, 0]` for three allowed keys. Softmax is approximately `[0.665, 0.245, 0.090]`. If scalar values are `[10, 0, -5]`, the output is:

\[
0.665(10)+0.245(0)+0.090(-5)=6.20.
\]

Attention does not copy only the top token; it blends all allowed values unless weights become nearly one-hot.

## Worked example 2 - Shapes

For batch \(B=2\), sequence \(n=4\), hidden size \(d=8\), and two heads:

- \(X\): `2 × 4 × 8`
- per-head \(Q,K,V\): `2 × 2 × 4 × 4` (batch, heads, sequence, head dimension)
- scores: `2 × 2 × 4 × 4`
- concatenated output: `2 × 4 × 8`

Always track dimensions; most implementation mistakes become visible there.

## Worked example 3 - Causal mask

For four tokens, the allowed matrix is:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

The state at token 2 cannot contain information from tokens 3 or 4. During training, targets can still be computed for all rows in one pass.

## Practice

1. Compute softmax approximately for scores `[0, 0]` and explain the weighted result.
2. Draw Q, K, V, score, mask, softmax, and V-mixing stages.
3. Explain why attention is contextual but an embedding lookup is not.
4. Compare encoder-only, decoder-only, and encoder-decoder models for sentiment, chat, and translation.
5. Explain what FlashAttention changes and what it does not change.

## Mastery check

You are ready when you can derive attention shapes, explain a modern pre-LN decoder block, and distinguish training parallelism from autoregressive decoding.

