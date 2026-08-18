---
name: bess-domain
description: Battery energy-storage vocabulary and where each concept lives in the GridSense corpus and feature set — SOH, DoD, C-rate, equivalent full cycles, calendar versus cycle ageing, thermal runaway. Use when writing golden-dataset items, interpreting an SOH prediction, or reading the ingested documents.
---

# BESS concepts as GridSense uses them

## The vocabulary

- **SOH — State of Health.** Present usable capacity as a percentage of rated capacity.
  What `predict_soh` returns. A number alone decides nothing: serviceability depends on the
  end-of-life threshold, which lives in the documentation, not in the model.
- **DoD — Depth of Discharge.** How far a cycle draws the pack down. **A fraction between 0
  and 1 throughout this repo, never a percentage.** Passing 70 where 0.7 was meant is
  accepted by the schema and produces a confidently wrong prediction.
- **C-rate.** Charge/discharge current relative to capacity. 1C discharges the pack in
  roughly an hour.
- **EFC — Equivalent Full Cycles.** Partial cycles normalised to full ones, so packs on
  different duty can be compared.
- **Calendar ageing vs cycle ageing.** Calendar ageing accrues with time regardless of use;
  cycle ageing accrues with throughput. Both are represented — `calendar_age_days` and
  `cycle_count` are separate model inputs for that reason.
- **Thermal runaway.** Self-sustaining exothermic failure. A safety topic, not a
  degradation one; it lives in the operations document.

## Where things live

Two documents, deliberately small:

- `data/docs/battery_soh.md` — SOH definitions, degradation mechanisms, and the factors
  that accelerate them. This is where end-of-life thresholds come from.
- `data/docs/bess_operations.md` — operating parameters, thermal management, thermal
  runaway, and the BMS.

The corpus is roughly six chunks. That is small enough that a retrieval metric can look
excellent for the wrong reason: retrieving the entire corpus is not retrieval. Keep `k`
well below the chunk count when interpreting recall.

## Derived features, and the skew to watch

Three features are derived rather than measured: `equivalent_full_cycles`,
`temperature_stress`, `calendar_stress`.

They are computed **twice** — in SQL at training time, and in pandas at serving time. If the
two ever diverge, the model is served inputs it never trained on: wrong predictions, no
error, nothing in the logs. There is currently no test comparing the two paths on the same
rows; they are kept identical by convention and code review. Treat any edit to either
formula as a change to both.

## Writing golden-dataset items

An item is `factual` or `synthesis` when answerable, and `vague`, `out_of_scope`, or
`insufficient_context` when not. `insufficient_context` means *in-domain but genuinely
absent from these two documents* — it is the category people get wrong, and an item filed
there that the corpus actually covers will make the refusal metric lie.

`must_contain` strings are verbatim substrings, checked against `data/docs` on every PR.
Write them from the document, not from memory.
