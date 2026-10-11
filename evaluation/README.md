# Evaluation

```bash
python -m evaluation.run_all        # about 10 minutes on a laptop CPU; writes results/SUMMARY.md
python -m unittest discover -s tests
```

| Script | What it measures |
|---|---|
| `benchmark_embeddings.py` | data linkage, representation quality, missing-view robustness, re-identification across club spells |
| `benchmark_roles.py` | role classification: the original engine re-examined, held-out comparison, cross-validation |
| `benchmark_speed.py` | query latency, original engines vs current |
| `benchmark_corpus.py` | multi-season corpus: held-out re-identification, thin vs rich data, identity metrics, roles, real transfers |
| `labelling.py` | builds a sheet for two human annotators and scores their agreement |

## Role labels and where they come from

| File | Players | Source | Status |
|---|---|---|---|
| `role_labels.csv` | 63 (58 outfield) | The project's original validation list plus the players named as prototype exemplars in the original role engine | The author's own labels |
| `role_labels_draft_unverified.csv` | 153 | Drafted by an AI assistant from general football knowledge, **before** any model prediction for these players was looked at | **Not verified by a human** |

The draft set exists because 58 labels are too few to measure anything precisely: the 95%
interval on an accuracy of 60% measured on 58 players is roughly ±13 points. It was written
blind so that it can serve as a held-out test for the original rule engine, whose weights
were tuned by hand against the original list.

Before any number that depends on the draft labels goes into a paper:

1. Run `python -m evaluation.labelling make`. It writes `labelling_sheet.csv` (333 regular
   starters, stratified by position, including every draft-labelled player) with two empty columns.
2. Two people fill in one column each, without looking at the model's output, the draft
   file, or each other's answers.
3. Run `python -m evaluation.labelling score`. It prints raw agreement and Cohen's kappa and
   writes the agreed labels to `role_labels_agreed.csv`.
4. Replace the draft file with the agreed labels and re-run `python -m evaluation.run_all`.

Three caveats that apply to the current role numbers:

- The draft labels come from a single annotator. A classifier trained and tested on them
  (the leave-one-out and cross-validation rows) partly learns that annotator's habits.
  The fairest comparison with the original engine is the one where the classifier is
  trained on the 58 own labels only.
- The position gate was revised (an RM/LM may now be an Inside Forward) after seeing that
  the original gate made the labelled role unreachable for 7% of players. Results are
  reported for both gates.
- The taxonomy has no full-back role. The 359 full-backs can only be "Ball Playing Defender"
  or "Wide Winger", and almost none are labelled.
