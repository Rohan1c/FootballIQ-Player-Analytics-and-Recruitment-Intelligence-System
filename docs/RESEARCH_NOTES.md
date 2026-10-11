# Research notes

Working notes for turning FootballIQ into a paper. Every number here is copied from
[`evaluation/results/SUMMARY.md`](../evaluation/results/SUMMARY.md), which is regenerated
by `python -m evaluation.run_all`. If the two ever disagree, the summary is right.

## 1. What the audit of version 1 found

| # | Weakness | Evidence |
|---|---|---|
| 1 | Reported role accuracy (80.4%) is inflated | 46 test players; 12 are also prototype exemplars, 5 are goalkeepers (correct by rule). On the other 29: 72.4%. Bonus weights and "confusion penalties" were tuned by hand on the same list. On 153 players it was never tuned on: 62.7% (95% CI 54.9-70.0). Chance within the position gate: 34%. |
| 2 | Prototype exemplars resolved by substring search | "Rodri" matched Rodrigo Bentancur, "Enzo" matched Lorenzo Colombo (a striker in the Deep Playmaker prototype), "Son" matched Brenden Aaronson, "Salah" matched Ibrahim Salah. 12 of 37 exemplars were not found at all. |
| 3 | Two role answers for one player | The role engine min-max scales its inputs; the compatibility, transfer, squad and XI engines passed it unscaled rows. The two paths disagreed for 35% of players. |
| 4 | The autoencoder did not compress | 63 inputs to a 64-unit ReLU bottleneck. Its embedding was worse than its own standardised inputs: position-group purity@10 of 0.572 against 0.600. |
| 5 | Playing-time leakage | Season totals (minutes, starts, goals) were inputs. Log-minutes can be predicted from the embedding with R² = 0.80. |
| 6 | Lossy, sometimes wrong merge | Exact-name join kept 2,057 of 2,690 players (82% of minutes), kept one club spell for mid-season movers, and matched 4 players to a namesake more than two years older or younger. |
| 7 | "Position-weighted" similarity had no effect | Every embedding was multiplied by the same constant, which cannot change a ranking. |
| 8 | Goalkeepers mixed with outfield players | For goalkeepers EA stores diving, handling and so on in the Pace, Shooting, ... columns. |
| 9 | Inefficiency and broken paths | A partner or replacement query took about 11 s because every candidate's role was recomputed per call. Three dashboard menu entries could not run (wrong argument types, a missing `save_outputs.py`). An XI "built around" a non-midfielder did not contain him. |
| 10 | README described data that is not there | xG, xA, progressive passes and pressures are not in the files. |

## 2. Method

**Record linkage.** Club spells are summed per player. Names are normalised (accents,
punctuation). Pass 1 accepts an exact normalised name if the club agrees, or, for
multi-word names, if the nation agrees. Pass 2 matches inside the same club using token
containment, abbreviation and surname rules. Pass 3 matches near-identical spellings
inside the same nation. Club and nation spellings differ between the sources
("Lazio" / "Latium", "us USA" / "United States"); the mapping is learnt by majority vote
from unambiguous matches. Every pass requires ages within two years and equal
goalkeeper status. Result: 2,366 players, 97.0% of minutes, no age gap above two years.

**Features.** For a count `c` over `n` 90s, the rate is estimated as
`(c + alpha) / (n + beta)` with a Gamma(alpha, beta) prior fitted by moments within each
FBref line (DF / MF / FW) on players with 10 or more 90s. Ratios use the Beta-Binomial
analogue. Rates are square-root transformed and z-scored on players with 450+ minutes.
The attribute view is 29 detailed EA attributes plus height and weight, z-scored.

**Representations compared.** Standardised features; PCA; a bottleneck autoencoder;
a masked multi-view autoencoder (random feature masking at 25%, a whole view dropped
with probability 0.3, mask indicators as extra inputs, reconstruction of all features
with each view weighted equally). Hyper-parameters were fixed before looking at any
label-based metric; a sweep over the embedding size is reported.

**Roles.** Multinomial logistic regression (C = 0.1) on the 44 features, then the
position gate.

## 3. Evaluation protocol

No ground truth exists for "similar players", so the protocol combines several partial ones.

| Test | Ground truth | What it shows |
|---|---|---|
| Position purity / linear probe | EA position (11 classes, and 7 side-agnostic families) | Does the space organise players by what they do? |
| Playing-time leakage | Log-minutes | Does the space encode how much a player played? (it should not) |
| Role retrieval (MAP) | Role labels, never used to fit the representation | Does the space separate tactical roles? |
| Seed stability | Agreement of neighbour sets across seeds | Is a neighbour list reproducible? |
| Missing view | A held-out player's own full embedding | Can a player with one source be placed correctly? |
| Re-identification | Mid-season movers have two FBref rows | Does a style description survive a change of club? |

All representation rows are scored on the same 1,248 outfield players (900+ minutes,
present in both the original and the new table). Stochastic models are run with five seeds.

## 4. Results

**Representation quality** (same 1,248 players):

| | Family purity@10 | Probe F1 | Minutes R² | Role MAP (own labels) |
|---|---|---|---|---|
| Original autoencoder embedding | 0.572 | 0.687 | 0.804 | 0.324 |
| Original 63 features | 0.600 | 0.750 | 0.854 | 0.352 |
| New features, standardised | 0.660 | 0.778 | 0.100 | 0.486 |
| New features, PCA-16 | 0.670 | 0.762 | 0.095 | 0.493 |
| New features, autoencoder-16 | 0.653 | 0.740 | 0.112 | 0.464 |
| New features, masked autoencoder-16 | 0.640 | 0.743 | 0.112 | 0.456 |

PCA-16 against the original embedding: +0.097 family purity (paired bootstrap 95% CI
0.086 to 0.109).

**Missing view** (five-fold, held-out players; share whose own full embedding is among
the 10 nearest):

| Available view | PCA + mean imputation | PCA + ridge imputation | Autoencoder | Masked autoencoder |
|---|---|---|---|---|
| Attributes only | 0.706 | **0.921** | 0.341 | 0.738 |
| Performance only | 0.141 | 0.295 | 0.083 | **0.441** |

**Re-identification** (38 movers, 76 queries, minutes-matched gallery; share of queries
whose other spell is in the nearest 5%):

| Features | Hit rate | 95% CI |
|---|---|---|
| Original season totals | 0.118 | 0.053-0.197 |
| Per-90, no shrinkage | 0.329 | 0.224-0.434 |
| Per-90 + empirical-Bayes shrinkage | 0.474 | 0.368-0.579 |

With an unmatched gallery of full-season players, season totals do as well as the shrunk
rates (0.487 against 0.566): both spells are half seasons, and totals pick that up. The
minutes-matched gallery removes that shortcut. Both are reported.

**Roles** (153 held-out players, draft labels):

| Classifier | Trained on | Accuracy | 95% CI |
|---|---|---|---|
| Chance within the gate | - | 0.342 | |
| Original rule engine | hand-tuned | 0.627 | 0.549-0.700 |
| Original rule engine, exemplars fixed | hand-tuned | 0.641 | 0.562-0.712 |
| Nearest prototype + gate | 58 own labels | 0.660 | 0.582-0.732 |
| Logistic regression + gate | 58 own labels | 0.621 | 0.542-0.699 |
| Logistic regression + gate, leave-one-out | 210 labels | 0.817 | 0.752-0.876 |

Ten-times-repeated five-fold cross-validation over all 211 labels: 0.778 with the revised
gate, 0.736 with the original gate, 0.765 when C is chosen by inner cross-validation.
Against the original engine on the 153 held-out players, the leave-one-out classifier is
right where the engine is wrong 44 times and wrong where it is right 15 times (exact
McNemar p = 0.0002).

**Latency.** Best-partners and replacement queries: from about 11-12 s each to under 5 ms
(exact timings depend on the machine; see section 6 of the summary).

## 5. How to read these results

1. **The improvement is in the data and the features.** Going from the original features
   to the new ones moves every metric; changing the model on top of the new features
   moves almost nothing.
2. **The autoencoder is not justified on this dataset.** PCA is at least as good on every
   full-data metric and is deterministic (the autoencoders agree with themselves across
   seeds on only 43-52% of neighbours). This is a negative result and should be
   presented as one.
3. **Masking matters when a view is missing.** A plain autoencoder fails when a view is
   absent (0.083 to 0.341); the masked one does not. But a linear baseline (ridge
   imputation + PCA) is better still when attributes are available, which is the case
   that matters for scouting outside the big five leagues.
4. **Hand-tuned role rules add nothing measurable.** Given the same 58 labels, a
   classifier with no hand-set weights matches the rule engine out of sample (62-66%
   against 63-64%). More labels help: accuracy rises to about 77% with 211.
5. **The originally reported 80.4% is not an out-of-sample figure.**

Something that was tried and dropped: a latent-consistency term that pulls the code of a
masked input towards the code of the full input. It diverged with the live encoder as
target and, once stabilised with a moving-average target, traded away reconstruction and
made every missing-view metric worse.

## 6. Threats to validity

- **Role labels.** 153 of the 211 are an unverified single-annotator draft. The
  leave-one-out and cross-validation figures train and test on that annotator's labels;
  the original engine was tuned to a different person's labels. The held-out rows that
  train on the 58 own labels are the like-for-like comparison.
- **The gate revision is post hoc.** Both gates are reported.
- **Position labels come from EA**, as does the attribute view. The performance-only row
  (family purity 0.543 from FBref statistics alone) is the cross-source evidence.
- **Shrinkage uses FBref's own position line** as the prior group, which leaks a coarse
  position signal into low-minute players. The main cohort is restricted to 900+ minutes,
  where the prior has little weight.
- **Re-identification has 38 players.** The intervals are wide.
- **One season, five leagues, and no passing or pressing data.**
- **Compatibility, trio and XI outputs are not evaluated** and should be described as
  rule-based applications.

## 7. Before submitting anywhere

1. Get the role labels verified by two annotators (`evaluation/README.md`), report
   Cohen's kappa, re-run the benchmarks. Aim for 300+ agreed labels.
2. Decide on a full-back role (or drop full-backs from the role task and say so).
3. Add passing and possession data if a source is available; this is the most likely
   way to raise role accuracy further.
4. If a second season can be added, test whether a player's embedding is stable from
   one season to the next. That is a much stronger version of the re-identification test.

## 8. A paper outline that the evidence supports

*Working title:* "What Actually Matters in Player-Similarity Systems: Linkage, Leakage
and Evaluation in Football Recruitment Analytics".

1. Introduction: similarity systems are easy to build and hard to evaluate.
2. Related work: player embeddings, role discovery, record linkage, empirical-Bayes rate estimation.
3. Data and linkage.
4. Features: two views, shrinkage, what is left out and why.
5. Evaluation protocol: the six tests of section 3 (the main contribution).
6. Experiments: section 4, including the two negative results.
7. Applications: global scouting, replacements, line-ups (qualitative).
8. Limitations and future work.

A claim the evidence does **not** support: that a deep model is what makes the system
work. Reviewers will check a PCA baseline first.
