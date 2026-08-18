---
name: docrag-triage
description: Work out why DocRAG refused or flagged an answer — which guardrail fired, what distinguishes the refusal reasons, and what each one implies about the corpus versus the model. Use when /ask or the ask_docs tool returns refused true, or an answer comes back with a medium or high uncertainty_level.
---

# Triaging a DocRAG refusal

Refusals are structural. `refused` is a boolean the code sets and `refusal_reason` names the
guardrail — never infer a refusal from the answer text, and never report one as a hedge.

## The refusal reasons, and what each tells you

Upstream — fired before any token was spent, so the corpus was never consulted:

| Reason | What happened | What it implies |
|---|---|---|
| `too_long` | Question exceeded the token cap | Rejected, never truncated — a truncated question is a different question |
| `prompt_injection` | Weighted pattern score cleared the threshold | The payload is redacted before it reaches the trace store |
| `out_of_scope` | Intent router matched a non-BESS request | The router is precision-tuned; anything ambiguous falls through to retrieval instead |

Retrieval and downstream:

| Reason | What happened | What it implies |
|---|---|---|
| `no_relevant_context` | Nothing cleared the relevance floor | The corpus probably lacks the answer |
| `low_confidence` | The model answered but below the confidence floor | The corpus likely lacks *enough* information |
| `invalid_output` | Every attempt failed schema validation | The corpus may well hold the answer — **retrying is reasonable** |
| `provider_error` | The call never reached a model at all | Auth, credit balance, rate limit or connection. **Retrying cannot help** — fix the configuration |

Two distinctions people get wrong here.

`invalid_output` is a model failure, not a corpus verdict: answering "I couldn't find that"
misreports the situation, because the corpus was never the problem.

`provider_error` vs `invalid_output` matters more. The first means there was no response to
parse; the second means there was one and it was unusable. They were merged once, and an
empty credit balance came back to users as *"the model's response was malformed. Please try
again."* — wrong about what happened, and advice that could never work. If you see
`provider_error`, check the key, the credit balance and the rate limit before touching
anything in the pipeline.

## Uncertainty is not refusal

A shaky answer is disclosed, not suppressed. Three signals feed the verdict — the model's
self-reported confidence, the best relevance score among the kept chunks, and the
deterministic groundedness ratio. One weak signal makes it `medium`, two or more `high`.

Carry `uncertainty_note` into whatever you write. Dropping it turns a flagged answer into
one that looks settled, which is the failure the disclosure exists to prevent.

The note lives in its own field rather than being prefixed onto `answer` so that it never
contaminates the text the eval scores for faithfulness. Do not merge them.

## Answers without citations

The prompt asks for sources and the schema does not enforce them. When the model returns
none, the chain falls back to citing every retrieved passage. That is a fallback, not
evidence the model actually named its sources — worth knowing before treating a citation
list as attribution.

## Two failure modes that look like something else

- **Fluent nonsense across every answer, plus retrieval that degrades for no reason.** On
  the offline Ollama path this is usually one cause, not two: a GPU that computes
  incorrectly returns confident garbage *and* writes corrupted embeddings, silently. Set
  `OLLAMA_NUM_GPU=0`, re-ingest, and re-check.
- **A guardrail blocking real questions.** A precision failure in the injection detector or
  the intent router is worse than no guardrail, because it is invisible from the outside.
  The upstream test suite asserts the false-positive side deliberately; keep it that way
  when adding a pattern.
