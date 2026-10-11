# FootballIQ: Player Analytics and Recruitment Intelligence System

A football scouting toolkit built on two independent descriptions of every player:
what he did on the pitch (FBref season statistics, big five European leagues, 2025-26)
and how he is rated (EA FC detailed attributes). It finds similar players, assigns
tactical roles, recommends replacements and partners, builds line-ups, and searches
about 12,200 players from other leagues for look-alikes.

It has two parts:

- **Live system (2025-26).** The season FBref still covers, but only with basic statistics
  since it removed its advanced data in January 2026.
- **Multi-season corpus (2017-18 to 2021-22).** 13,229 player-seasons with passing,
  carrying, pressing and shot-creation data, used to learn and test the similarity
  metric properly and to power a historical explorer.

Version 2 is a rebuild of the original internship project around a measured audit of
its weaknesses. Every claim below is backed by a script in `evaluation/`; the full
tables are in [`evaluation/results/SUMMARY.md`](evaluation/results/SUMMARY.md) and the
reasoning is in [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md).

---

## What changed from version 1, in numbers

| | Version 1 | Version 2 |
|---|---|---|
| Players kept by the FBref-EA merge | 2,057 (82% of minutes) | 2,366 (97% of minutes) |
| Playing time predictable from the embedding (R², lower is better) | 0.80 | 0.10 |
| Nearest neighbours in the same position group | 57% | 67% |
| Same-role retrieval, mean average precision (58 own labels) | 0.32 | 0.49 |
| Role accuracy on players the engine was not tuned on (153 draft labels) | 63% | 62-66% trained on the same 58 labels; about 77% cross-validated with all 211 |
| "Best partners" or "replacements" query | about 11 s | under 5 ms |

The 80.4% role accuracy reported for version 1 was measured on 46 players, 12 of which
were also the prototype exemplars and 5 of which were goalkeepers (correct by rule).
See section 5 of the summary for the breakdown.

## What better data buys (multi-season corpus)

All figures are on players the models never saw during fitting (five random player splits).
"Re-identification" asks: given a player's season, is his own next season the nearest of
about 1,500 candidates?

| | Statistics the 2025-26 file has (13) | Rich statistics (83) | Rich + identity metric |
|---|---|---|---|
| Next season is the single nearest | 6% | 45% | **57%** |
| Next season is in the nearest 10 | 25% | 77% | **86%** |
| Same, for players who changed club (nearest 10) | 20% | 57% | **70%** |
| Neighbours in the same position group | 64% | 72% | 71% |
| Role accuracy from statistics alone (153 labelled players) | 53% | 64% | - |

The identity metric is linear discriminant analysis with players as classes: it learns from
several seasons of the same player which statistics are stable and which are noise.

Three findings are negative and are reported as such:

- **The autoencoder does not beat PCA on this data.** With 44 features and about 1,800
  training players, a 16-dimension PCA matches or beats both autoencoder variants on
  every measure of embedding quality. The masked autoencoder wins in one setting only:
  embedding a player from match statistics alone. PCA is therefore the default; the
  masked autoencoder is kept as an option (`EMBEDDING_METHOD = "mdae"` in
  `footballiq/config.py`).
- **Deep models do not beat linear ones on the corpus either.** A linear identity metric
  reaches 57% top-1 re-identification; contrastive networks with one and two hidden
  layers reach 46% and 41%, and autoencoders re-identify fewer players than the plain
  standardised features do (41-42% against 45%).
  About 1,800 training players are not enough for the extra capacity.
- **Clubs' real signings resemble the players they replace only slightly.** Measured on
  236 club-summers, a signing sits at the 45th-46th percentile of the market by similarity
  to the departed player (50 would be no relation), whichever representation is used.
- **The gains come from data handling and feature design**, not from the model:
  record linkage, per-90 rates with empirical-Bayes shrinkage, removing availability
  columns, and treating goalkeepers separately.

---

## Pipeline

```
FBref season stats            EA FC attributes
        |                            |
        +------- record linkage -----+        footballiq/linkage.py
                      |
        +-------------+--------------+
        |                            |
performance view (13)        attribute view (31)   footballiq/features.py
per-90, EB-shrunk            detailed ratings
        |                            |
        +-------- embedding ---------+        footballiq/embedding.py  (PCA-16 | masked AE)
                      |
   shared PlayerStore (features, embedding, roles)   footballiq/store.py
                      |
  similarity | roles | compatibility | transfer | squad | scouting | report
```

### Data linkage (`footballiq/linkage.py`)

The original exact-name join lost 23% of players (Vinícius Júnior, Zubimendi and Gabriel
among them), kept one club spell for mid-season movers, and matched a few players to a
different person with the same name. The linkage now:

1. sums a player's club spells into one season row;
2. matches on an accent-insensitive name, confirmed by club or nation;
3. matches within the same club on nicknames and short names ("Vini Jr.", "Gabriel");
4. always requires the two sources to agree on age (within 2 years) and on goalkeeper status.

### Features (`footballiq/features.py`)

- **Performance view**: 11 per-90 rates and 2 ratios, each shrunk towards the positional
  average in proportion to how little the player has played (Gamma-Poisson and
  Beta-Binomial empirical Bayes).
- **Attribute view**: 29 detailed EA attributes plus height and weight. The six "face"
  ratings are dropped because they are weighted sums of the detailed ones.
- Minutes, starts, points per match and plus/minus are left out: they describe
  availability and team strength, not style.
- Goalkeepers get their own feature set and their own similarity space.

### Roles (`footballiq/roles.py`)

A regularised logistic regression on the 44 features, followed by the position gate
(a centre-back cannot be a Poacher). It replaces the hand-written bonus and penalty
weights, so its accuracy can be cross-validated.

| Category | Roles |
|---|---|
| Midfield | Deep Playmaker, Creative Playmaker, Ball Winner, Box-to-Box |
| Attack | Wide Winger, Creative Winger, Inside Forward, False 9, Poacher, Target Forward |
| Defence | Ball Playing Defender, Defensive Defender |
| Goalkeeper | Goalkeeper |

### Engines

| Module | What it does |
|---|---|
| `similarity.py` | Most similar players in the same position group, with the features they share |
| `compatibility.py` | Pairwise tactical compatibility (original formula, vectorised) |
| `transfer.py` | Replacement candidates by style, role, position and age |
| `squad.py` | Best midfield trio; 4-3-3 built around any player |
| `scouting.py` | **New.** Look-alikes among EA players outside the big five leagues |
| `report.py` | One-page report combining all of the above |
| `history.py` | **New.** Historical explorer on the 2017-22 corpus: similar player-seasons, value and age filters, style timeline |
| `corpus.py`, `metric.py` | Corpus features (83 per-90 statistics) and the identity metrics learnt from them |
| `charts.py` | Similarity, role, heatmap, network and embedding-map figures |

**Global scouting** works because the embedding can be computed from attributes alone:
the missing performance view is predicted from the attributes by ridge regression.
For held-out players, the attribute-only embedding has the player's own full embedding
among its 10 nearest in 92% of cases (summary, section 3).

```
Players outside the big five leagues most similar to Pedri, aged 23 or under:
 1. Christos Zafeiris     CM   age 22  Slavia Praha
 2. Kian Fitz-Jim         CM   age 22  Ajax
 3. Mohammed Diomande     CDM  age 23  Rangers
```

**Historical explorer** uses the corpus, where the statistics are rich enough to separate
styles of passing and pressing:

```
Most similar to Kevin De Bruyne in 2021-22 (Manchester City, Attacking Midfield):
 1. Bruno Fernandes     2021-22  Manchester Utd   shared: ThroughBalls, PassPenArea, KeyPasses
 2. Hakan Çalhanoğlu    2020-21  Milan            shared: Corners, KeyPasses, SCA
 3. Mason Mount         2021-22  Chelsea          shared: Corners, Ast, PassPenArea
```

---

## Getting started

```bash
pip install -r requirements.txt

python footballiq_dashboard.py              # interactive menu
python save_outputs.py --player "Pedri"     # write all charts to outputs/

python -m footballiq.linkage                # rebuild data/players_linked.csv from the raw files
python -m unittest discover -s tests        # 34 tests
python -m evaluation.run_all                # all benchmarks, about 10 minutes
```

From Python:

```python
from footballiq.similarity import similar_players
from footballiq.transfer import find_replacements
from footballiq.scouting import global_lookalikes

similar_players("Pedri", top_n=5)
find_replacements("Harry Kane", younger_only=True)
global_lookalikes("Lamine Yamal", max_age=21)

from footballiq.history import similar_seasons, style_timeline
similar_seasons("Mohamed Salah", season=2018, same_season=True, max_value_eur=20e6)
style_timeline("Joshua Kimmich")
```

Dependencies are NumPy, pandas, scikit-learn, SciPy, Matplotlib, seaborn and NetworkX.
The masked autoencoder is written in NumPy, so no deep-learning framework is needed.

---

## Repository layout

```
footballiq/            core package (see the tables above)
footballiq_dashboard.py  terminal menu
save_outputs.py        writes every chart
data/                  raw FBref and EA files, and the linked table
data/corpus/           multi-season corpus (see data/corpus/README.md for the source)
scripts/               build_corpus.py: rebuilds the corpus from its public source
evaluation/            benchmarks, role labels, results/
tests/                 unit and regression tests
docs/RESEARCH_NOTES.md findings, method, limitations, paper outline
legacy/                version 1, kept runnable as the benchmark baseline
outputs/               generated figures
```

---

## Limitations

- **Role labels are few and partly unverified.** 58 are the author's; 153 were drafted
  by an AI assistant and must be checked by two human annotators before publication
  (`evaluation/README.md` explains how).
- **The live season has thin statistics.** FBref removed its advanced data in January
  2026, so the 2025-26 tables have shots, goals, crosses, fouls, tackles won and
  interceptions, but no passing, carrying or pressing data. The corpus shows what that
  costs (table above). Current players cannot be given rich statistics from a free source.
- **The corpus ends in 2021-22** and has no EA attributes, so it is a research and
  history tool, not a description of today's players.
- **EA attributes are expert ratings**, not measurements.
- **Compatibility, trio and XI scores are heuristics.** Their weights are design choices;
  there is no ground truth for "chemistry" in the data, so they are not validated.
- **No full-back role.** Full-backs are forced into "Ball Playing Defender" or "Wide Winger".
- Five leagues only.

---

## License

MIT License

## Author

**Rohan Srinivas Ponnana**
CSE Artificial Intelligence & Machine Learning
Football Analytics · Representation Learning · Recruitment Intelligence
