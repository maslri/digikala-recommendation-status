# 07 - Prompting and In-Context Learning

## Goal

Treat prompts as versioned, evaluated interfaces. Learn zero/few-shot prompting, templates, roles, structured output, grounded tasks, reasoning, and systematic debugging.

## Page-by-page lesson

### Page 1 - Prompt as interface

A prompt supplies task, context, examples, and output expectations to a probabilistic model. It is an input specification, not a spell and not a hard guarantee.

### Page 2 - Four questions

Ask what changes, what belongs in a strong prompt, how examples influence behavior, and when prompting is no longer the correct adaptation lever.

### Page 3 - Adaptation levers

Prompting changes immediate context; RAG adds external evidence; tools add computation/action; fine-tuning changes persistent behavior. Choose based on the diagnosed missing ingredient.

### Page 4 - In-context learning (ICL)

Weights \(\theta\) remain fixed. The model conditions on instructions and examples in the current sequence. Once the context is removed, that temporary behavior is not stored in parameters.

### Page 5 - Six jobs

A robust prompt separates role/policy, task, input context, demonstrations, constraints, and output format. Delimit variable/untrusted text so it cannot be confused with instructions.

### Page 6 - Zero-shot and few-shot

Zero-shot uses instructions only and is the baseline. One-shot or few-shot adds demonstrations when labels, style, edge cases, or output shape are difficult to specify abstractly. Examples consume context and can introduce bias.

### Page 7 - Capability and scale

ICL effectiveness differs across models and tasks. A prompt pattern that worked in one study is not universal. Validate with the exact deployed model and version.

### Page 8 - What demonstrations communicate

Examples reveal input distribution, label semantics, transformation pattern, tone, and format. Good examples are correct, diverse, representative, and focused on real decision boundaries.

### Page 9 - Random-label finding

Some experiments found demonstrations remained useful even with incorrect labels, suggesting format and input distribution can drive part of ICL. Do not conclude labels are unimportant; correct targets remain essential in production prompting and evaluation.

### Page 10 - Selection and order sensitivity

Which examples appear, their similarity to the query, class balance, and ordering can change results. Store demonstration sets as versioned prompt configuration and test permutations where risk matters.

### Page 11 - Prompt templates

Keep stable instructions separate from variables. Escape or clearly delimit inserted content. Version the template, model, examples, decoding settings, and schema together.

### Page 12 - Roles and personas

System/developer/user roles define instruction authority in chat interfaces. Persona cues such as “act as an editor” steer style or perspective but do not grant factual expertise or security authority.

### Page 13 - Structured output

Specify field names, types, enums, required fields, and missing-value behavior. Downstream code needs a contract such as JSON Schema, not merely “answer as JSON.”

### Page 14 - Reliability layers

Prompt-only formatting can break. Production systems use constrained decoding or structured-output APIs, parsing, schema validation, semantic validation, retries, and safe fallback.

### Page 15 - Classification

Define labels by inclusion and exclusion rules. Add boundary cases, permit `uncertain` when appropriate, and request only the machine-consumed fields. Measure confusion by class, not only average accuracy.

### Page 16 - Extraction

Make absence legal: return `null` when unsupported. Require exact source spans when auditability matters. Separate normalization (for example date format) from extraction and validate both.

### Page 17 - Summarization

Define audience, purpose, length, preservation requirements, forbidden additions, and handling of uncertainty. A summary for an executive and one for an engineer optimize different information.

### Page 18 - Grounded QA

Provide sources and an answerability rule: use only supplied context; if it does not support an answer, say so. Ask for citations linking claims to passages and test unanswerable cases.

### Page 19 - Intermediate reasoning

Worked examples can improve multi-step performance by demonstrating decomposition. For production, prefer concise verifiable intermediate artifacts—calculations, tool calls, plans, tests—over trusting an unconstrained rationale.

### Page 20 - Unfaithful explanations

A plausible explanation may not accurately reflect the computation that produced the answer. Treat rationale as generated content requiring verification, not privileged access to internal reasoning.

### Page 21 - Failure taxonomy

Common failures are ambiguous task, missing context, poor demonstrations, conflicting instructions, invalid format, and model capability limits. Name the category before changing wording.

### Page 22 - Evaluation loop

Build representative cases, define metrics/rubrics, run the prompt, inspect failures, change one factor, and rerun. Avoid optimizing one memorable example at the expense of the distribution.

### Page 23 - When prompting is insufficient

Use RAG for changing/private/sourceable facts, tools for exact computation or action, fine-tuning for stable repeated behavior gaps, and deterministic code for hard constraints.

### Page 24 - Repair exercise

“Read our policy and answer accurately” lacks policy text, answerability, conflict/version policy, citations, audience, and output expectations. The next lever is retrieval if the policy is external.

### Page 25 - Sources

Separate research findings from current API guidance. Prompt behavior is model- and version-dependent, so current official documentation and local evaluation both matter.

## Worked example - Classification prompt

```text
Task: Classify one support message.
Labels:
- refund: asks to return money already paid
- cancel: asks to stop an unfulfilled order
- status: asks where an order is
- other: none of the above

Rules:
- Choose exactly one label.
- If both refund and cancel appear, choose refund only when payment reversal is requested.

Input (untrusted data):
<message>{{MESSAGE}}</message>

Return JSON matching:
{"label":"refund|cancel|status|other"}
```

The label definitions and tie-break rule are more useful than a persona such as “you are an expert classifier.”

## Worked example - Grounded extraction

```text
Extract invoice_number and due_date only from SOURCE.
If a field is absent, return null. Do not infer it.
For each non-null field, include the exact supporting quote.

SOURCE:
Invoice AC-104. Payment must arrive by 2026-09-03.
```

Expected fields are `AC-104` and `2026-09-03`; a validator then checks date format and that quotes occur in the source.

## Practice

1. Rewrite a vague summarization prompt with audience, length, and preservation rules.
2. Create three few-shot examples around a difficult label boundary.
3. Design five evaluation cases including missing data and prompt injection inside input text.
4. For four failures, choose prompt, RAG, tool, fine-tune, or deterministic code.

## Mastery check

You are ready when you can create a delimited, schema-driven prompt, evaluate it on fixed cases, and recognize when a different adaptation lever is needed.

