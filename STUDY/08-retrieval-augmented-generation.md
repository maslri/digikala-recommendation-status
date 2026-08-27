# 08 - Retrieval-Augmented Generation (RAG)

## Goal

Build and debug a pipeline that retrieves external evidence, assembles context, generates grounded answers, and evaluates retrieval separately from generation.

## Page-by-page lesson

### Page 1 - RAG mental model

RAG supplies fresh, private, replaceable, and sourceable evidence at inference time. It changes what the model sees in this request, not what its weights know.

### Page 2 - Four questions

Ask why to retrieve, what the search unit is, how evidence reaches the generator, and how each pipeline stage will be evaluated.

### Page 3 - Adaptation choice

Prompt for specification, RAG for evidence, tools for computation/action, and tuning for persistent behavior. RAG is not a remedy for every reasoning or formatting problem.

### Page 4 - Fixed parameters

The generator computes \(P(answer\mid question,retrieved\ context;\theta)\) with fixed \(\theta\). Updating the index can change accessible knowledge without retraining the model.

### Page 5 - Why external evidence

Retrieval is valuable when information is current, private, frequently updated, too large for a prompt, or must be cited. It also permits document-level access control and deletion.

### Page 6 - Pipeline stages

Ingest, parse, clean, chunk, enrich with metadata, embed/index, formulate query, retrieve, filter/rerank, assemble context, generate, cite, and evaluate. A final wrong answer may originate at any earlier stage.

### Page 7 - Basic pipeline

The online core is question → query representation → index search → top candidates → context → generator → answer. Production versions add permissions, caching, reranking, observability, and fallback.

### Page 8 - Offline versus online

Offline ingestion determines corpus quality, chunk boundaries, metadata, embeddings, and index. Online answering determines query rewriting, filters, top-k, reranking, context ordering, and generation.

### Page 9 - Parametric and non-parametric memory

The model's weights store compressed patterns; the retrieval corpus stores replaceable records. RAG combines them, but retrieved content is evidence only if it is relevant and trustworthy.

### Page 10 - Chunks are search units

Retrievers usually rank chunks, not whole documents. A chunk must retain enough identity and meaning to match a query and later support a citation.

### Page 11 - Chunk-size trade-off

Small chunks are precise but lose context; large chunks preserve context but dilute embeddings and waste prompt space. Tune by document structure, question type, retriever, and model—not a universal token number.

### Page 12 - Boundaries, overlap, metadata

Semantic boundaries avoid splitting units; overlap protects cross-boundary facts; metadata supports filtering, identity, versions, dates, and access control. These controls solve different problems.

### Page 13 - Contextual ambiguity

“It is valid for 30 days” is useless without the policy/product identity. Prepend section/document context or create contextualized chunks so local text remains interpretable.

### Page 14 - Dense retrieval

An encoder maps queries and passages into vectors. Similarity, often dot product or cosine, ranks candidates. Training should make relevant query-passage pairs closer than irrelevant ones.

### Page 15 - Lexical, dense, hybrid

Lexical methods such as BM25 excel at exact identifiers and rare terms. Dense retrieval handles semantic paraphrase. Hybrid search combines scores or candidate sets to cover both signals.

### Page 16 - Top-k

Increasing k can improve recall but adds noise, duplicated evidence, latency, and context cost. Select k using downstream evidence recall and answer performance.

### Page 17 - Vector index is one layer

Approximate nearest-neighbor search accelerates vector lookup; it does not parse documents, enforce permissions, rerank, assemble context, generate, or verify citations.

### Page 18 - Reranking

A fast first-stage retriever produces many candidates. A slower cross-encoder or LLM-based reranker jointly scores each query-chunk pair, then forwards a smaller high-quality set.

### Page 19 - Two quality axes

Retrieval may succeed while the generator ignores evidence; generation cannot recover evidence never retrieved. Evaluate context relevance/recall and answer faithfulness/correctness separately.

### Page 20 - Context assembly

Deduplicate, group related chunks, preserve source metadata, order evidence, fit the token budget, mark boundaries, and state grounding/citation rules. Blind concatenation is not enough.

### Page 21 - Lost in the middle

Some models use evidence less reliably when it appears in the middle of long contexts. Keep context focused, rank strongly, and test evidence placement rather than assuming longer is safer.

### Page 22 - Citation quality

Citation recall asks whether claims needing support are cited. Citation precision/entailment asks whether the cited passage actually supports the claim. A citation can look credible and still be irrelevant.

### Page 23 - Conflicting sources

Use version and effective-date metadata. Define authority and conflict policy: prefer current approved policy, disclose conflict, or refuse pending clarification. Do not let the model silently blend incompatible rules.

### Page 24 - Failure taxonomy

Corpus failure: answer absent. Retrieval failure: present but not returned. Context failure: returned but dropped/obscured. Generation failure: evidence present but answer wrong. Citation failure: unsupported mapping.

### Page 25 - First broken stage

If the correct policy exists but its chunk lacks the product name, repair chunk identity before tuning generation. Fix the earliest failing stage because downstream changes cannot recover missing evidence reliably.

### Page 26 - Layered evaluation

Audit corpus answerability, parser/chunk integrity, retrieval recall/ranking, context inclusion, answer correctness/faithfulness, citations, latency, cost, and access-control behavior.

### Page 27 - Retrieval metrics

Recall@k asks whether relevant evidence appears in top k. Precision@k measures relevant fraction. MRR emphasizes the first relevant rank. nDCG rewards graded relevance with higher ranks. Choose the metric matching evidence consumption.

### Page 28 - Answer, grounding, citations

Correctness compares with the desired answer; faithfulness tests whether claims are supported by context; citation quality tests claim-source links. They can disagree and require separate labels.

### Page 29 - Product evaluation set

Include ordinary, exact-ID, paraphrase, cross-section, unanswerable, stale/conflicting, multilingual, permission-denied, and adversarial cases. Preserve a release gate independent of prompt tuning examples.

### Page 30 - Four design questions

Why retrieve? What is indexed? How is context assembled? How is each stage measured? Every RAG decision should map to one of these questions.

### Page 31 - Sources

Use original RAG/retrieval papers for mechanisms, benchmark papers for metric definitions, and current system documentation for implementation-specific behavior.

## Worked example 1 - Cosine similarity

For query vector \(q=[1,1]\), passage A `[1,0]`, passage B `[1,1]`:

\[
\cos(q,A)=1/\sqrt2\approx0.707,\qquad \cos(q,B)=1.
\]

B ranks higher. Real embeddings have hundreds or thousands of dimensions; similar vectors do not guarantee factual relevance.

## Worked example 2 - Retrieval metrics

Top five relevance labels are `[0,1,0,1,0]` with two relevant chunks total.

- Recall@3 = 1/2 = 0.5.
- Precision@3 = 1/3.
- Reciprocal rank = 1/2 because first relevant is rank 2.

These metrics expose different ranking behavior.

## Minimal RAG pseudocode

```python
chunks = retrieve(query, filters={"tenant": tenant_id}, k=20)
chunks = rerank(query, chunks)[:5]
context = assemble(chunks, token_budget=3000)
answer = generate(question=query, context=context,
                  rule="Use only context; cite every factual claim; say insufficient when needed")
validate_citations(answer, chunks)
```

Permissions must be enforced in trusted retrieval code, not left to the prompt.

## Practice

1. Chunk a policy with headings, tables, and exceptions; preserve identity metadata.
2. Design a hybrid retrieval case involving product ID `ZX-4812` and a semantic paraphrase.
3. Create ten evaluation questions covering the failure taxonomy.
4. For a wrong answer, write tests that isolate corpus, retrieval, context, and generation.
5. Define conflict handling for two policies with different effective dates.

## Mastery check

You are ready when you can locate the first broken stage, calculate ranking metrics, and design citations and access control as system features rather than prompt wishes.

