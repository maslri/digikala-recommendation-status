# 06 - Systems and Efficiency for LLMs

## Goal

Diagnose memory and speed bottlenecks and choose precision, checkpointing, attention, cache, batching, or parallelism techniques that address the actual cause.

## Page-by-page lesson

### Page 1 - Bottleneck-first thinking

Optimizations target different resources. First classify the workload (training, prefill, decode) and bottleneck (capacity, bandwidth, arithmetic, communication, or scheduler).

### Page 2 - Six questions

The session covers decode behavior, memory accounting, precision/quantization, attention/cache, batching, and distributed parallelism.

### Page 3 - 7B model on 24 GB

BF16 weights alone need roughly 14 GB, but runtime overhead, activations, temporary buffers, and KV cache can exceed the remaining memory. For training, gradients and optimizer states make the gap much larger.

### Page 4 - Decode is often bandwidth-bound

At small batch sizes, each generated token reads many model weights but performs too little reuse to saturate compute. Memory bandwidth, not theoretical FLOPs, can dominate latency.

### Page 5 - Memory hierarchy

Registers/shared SRAM are small and fast; high-bandwidth GPU memory is large but slower; host memory and storage are farther away. Efficient kernels reuse tiles near compute and avoid repeated HBM traffic.

### Page 6 - Training versus inference tensors

Training retains weights, gradients, optimizer states, activations, and buffers. Inference retains weights, temporary activations, and KV cache. Their memory formulas and useful optimizations differ.

### Page 7 - Training-state estimate

A common mixed-precision Adam estimate is around 16 bytes per trainable parameter: low-precision weights, gradients, FP32 master weights, and two FP32 moments. Seven billion parameters can therefore exceed 100 GB before activations.

### Page 8 - Activation memory

Activations grow with batch, sequence length, layers, hidden dimensions, and saved intermediates. Long sequences can trigger OOM even when weights fit. Inspect peak allocated memory rather than checkpoint size.

### Page 9 - Activation checkpointing

Checkpointing saves selected boundary activations and recomputes missing intermediates during backward. It lowers memory while increasing arithmetic and training time.

### Page 10 - FP16 versus BF16

Both use 16 bits. FP16 has more fraction precision but a much smaller exponent range; BF16 has FP32-like range with fewer fraction bits. BF16 often reduces overflow/underflow risk on supported hardware.

### Page 11 - Mixed precision

Storage, matrix multiplication inputs, accumulation, reductions, and optimizer state may use different formats. Numerically sensitive operations can remain higher precision. A recipe requires kernel and hardware support.

### Page 12 - Quantization target

Weight quantization reduces checkpoint bytes and weight bandwidth. Activations, cache, and accumulators may remain 16-bit or higher. Training-aware and post-training methods have different requirements.

### Page 13 - Weight-size arithmetic

Raw size is parameters × bits/8. A 7B model is about 14 GB at 16-bit, 7 GB at 8-bit, and 3.5 GB at 4-bit, plus scales, zero points, metadata, unquantized layers, and packaging.

### Page 14 - LLM.int8 outliers

Rare large activation dimensions can suffer badly under uniform INT8 scaling. LLM.int8 separates outlier features into a higher-precision path while handling most computation in INT8.

### Page 15 - GPTQ, AWQ, bitsandbytes

GPTQ is a post-training weight-quantization approach using calibration and reconstruction. AWQ protects salient weights using activation information. bitsandbytes is a software library offering quantized kernels and training utilities. Names can refer to algorithm, format, or runtime path.

### Page 16 - Size versus speed

Compressed weights help latency only if loading, dequantization, matrix kernels, batching, and hardware exploit them. A 4-bit model can be slower than optimized BF16 on an unsuitable stack.

### Page 17 - Standard attention traffic

Naive implementations materialize score and probability matrices in HBM, causing large reads/writes in addition to arithmetic. Memory traffic can be the limiting factor.

### Page 18 - FlashAttention

FlashAttention tiles Q/K/V and fuses operations so intermediate matrices need not be stored in full. It computes exact dense attention up to normal numerical differences; it does not remove quadratic pairwise arithmetic.

### Page 19 - Prefill versus decode

Prefill processes the prompt with large parallel matrix operations and creates the cache. Decode processes one new position per sequence repeatedly, relying on cache and often suffering lower hardware utilization.

### Page 20 - KV-cache growth

A rough cache formula is `2 × layers × tokens × KV heads × head_dim × bytes × concurrent sequences`; factor 2 is K and V. Cache is persistent for active requests.

### Page 21 - MHA, GQA, MQA

Multi-head attention has many Q and K/V heads. GQA shares K/V across groups of query heads. MQA shares one K/V head. Fewer K/V heads reduce cache and decode bandwidth while preserving multiple query heads.

### Page 22 - Long-context costs

Longer context increases prefill attention arithmetic roughly quadratically, KV cache linearly, and decode attention work per generated token roughly linearly in existing context. FlashAttention addresses IO, not all three costs.

### Page 23 - Batching

Batching reuses weight loads and improves throughput but increases queueing, cache use, and per-request latency. Continuous batching admits and removes sequences dynamically instead of waiting for a fixed batch to finish.

### Page 24 - PagedAttention

Paged KV-cache management allocates blocks on demand rather than reserving one contiguous maximum-length region per request. This reduces fragmentation and enables larger useful batches.

### Page 25 - Parallelism types

Data parallelism replicates a model and splits batches. Tensor parallelism splits large layer operations across devices. Pipeline parallelism assigns layer stages to devices and schedules microbatches. Each introduces distinct communication.

### Page 26 - ZeRO and FSDP

Standard data parallelism replicates model states. ZeRO/FSDP shard optimizer states, then gradients, then parameters at stronger stages, trading memory savings for communication and implementation complexity.

### Page 27 - Multi-dimensional parallelism

Large runs combine data × tensor × pipeline dimensions. The product is world size. Map high-communication groups to fast interconnects and account for pipeline bubbles and imbalance.

### Page 28 - Diagnosis table

Weight-capacity OOM suggests quantization/sharding; activation OOM suggests shorter sequences, microbatch reduction, or checkpointing; cache OOM suggests GQA/MQA, paging, fewer concurrent tokens, or lower cache precision; low decode throughput suggests batching and kernels.

### Page 29 - Sources

Verify techniques using the original systems papers and the exact runtime's documentation/benchmarks. Performance claims are hardware, shape, and version dependent.

## Worked example 1 - Memory

For 7B parameters:

- BF16 inference weights: \(7B\times2\approx14\) GB decimal.
- 4-bit raw weights: \(7B\times0.5\approx3.5\) GB.
- Mixed-precision Adam training state at 16 bytes/parameter: \(112\) GB before activations.

This explains why “the checkpoint fits” does not imply “training fits.”

## Worked example 2 - KV cache

Use 32 layers, 4 KV heads, head dimension 128, 8,192 tokens, BF16 (2 bytes), one sequence:

\[
2\times32\times4\times128\times8192\times2\approx536{,}870{,}912\text{ bytes},
\]

about 512 MiB per sequence. Ten concurrent long requests need roughly 5 GiB just for cache.

## Diagnostic mini-lab

```text
Symptom: OOM only during backward
Likely: saved activations or training states
Test: lower microbatch/sequence; inspect peak by phase
First tools: activation checkpointing, sharding

Symptom: low tokens/s at batch 1 but much better at batch 16
Likely: weight-bandwidth/utilization limit
First tools: continuous batching, optimized kernels, suitable quantization
```

## Practice

1. Calculate raw sizes for a 13B model at 16, 8, and 4 bits.
2. Double context in the KV example and state which costs double or quadruple.
3. Choose an optimization for weight OOM, activation OOM, and cache OOM.
4. Explain why FlashAttention and quantization solve different problems.
5. Compare data, tensor, and pipeline parallelism using “what is split?”

## Mastery check

You are ready when you diagnose first, calculate approximate memory, and explain why an optimization that reduces capacity may not improve latency.

