# Clean Social-Label Control

This add-on replaces the confounded mixed-group interpretation of the legacy
Exp5 with a prespecified label-only contrast.  The primary comparison is:

- `neutral_source_label`: `Source 1: B` ... `Source 5: B`
- `model_source_label`: `Model 1: B` ... `Model 5: B`

Every other prompt component is identical.  A tokenizer preflight selects a
generic neutral label whose complete chat prompt has the same token count as
the `Model` prompt on every selected item.  `unlabelled_numbered` is retained as
a descriptive secondary control, not as the primary causal comparison.

Formal runs must replay an archived private baseline with
`--reference-jsonl`.  The runner validates item IDs, question, options, gold
answer, distractor, and the parsability of the recorded baseline before making
the first final-answer request.

Offline checks:

```bash
python experiments/2026-08-29_clean_social_label/test_conditions.py
```

After a run, analyze the primary contrast with
`analysis/paired_protocol_analysis.py` using a manifest whose comparison is
`["neutral_source_label", "model_source_label"]`.
