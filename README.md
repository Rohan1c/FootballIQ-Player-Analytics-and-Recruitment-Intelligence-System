# FootballIQ: Player Analytics and Recruitment Intelligence System

An AI-powered football intelligence platform combining autoencoder-based representation learning, role archetype discovery, similarity modeling, compatibility analysis, transfer recommendations, and tactical squad construction using FBref statistics and FIFA attributes (~2058 players).

---

## Project Overview

FootballIQ is a research-grade football analytics and scouting platform designed to assist clubs, analysts, and recruiters in evaluating players, discovering replacements, building squads, and analyzing tactical fit.

The system integrates statistical performance data from FBref with FIFA player attributes to produce actionable football intelligence across seven interconnected modules — all accessible through a single interactive dashboard.

---

## System Architecture

```
Raw Data (FBref + FIFA)
        |
Data Preprocessing & Merging
        |
Autoencoder Training
        |
Latent Embeddings (latent_embeddings.npy)
        |
  ______________________________
  |        |        |          |
Similarity  Role   Compat.  Transfer
Engine    Engine   Engine    Engine
  |________|________|__________|
                |
        Squad Builder
        Starting XI Builder
        Intelligence Reports
```

---

## Dataset

Merged dataset of ~2057 players combining:

**FBref Statistics**
- Goals, assists, xG, xA
- Passing metrics (progressive passes, key passes, pass completion)
- Defensive actions (tackles, interceptions, pressures)
- Progressive carries and passing distance
- Playing time and 90s played

**FIFA Attributes**
- Technical: pace, shooting, passing, dribbling, defending, physicality
- Mental: vision, positioning, aggression, composure
- Physical: strength, stamina, balance, jumping

Position column (`Position`) contains FIFA-style labels:
`GK CB RB LB CDM CM CAM RM LM RW LW ST`

---

## Machine Learning Pipeline

### Autoencoder

A feedforward autoencoder compresses high-dimensional player feature vectors into a compact latent space.

- **Input**: normalized player stat + attribute vector
- **Latent space**: lower-dimensional player embedding
- **Output**: `models/latent_embeddings.npy`

The latent embeddings capture player style, tendencies, and hidden football characteristics beyond raw statistics — enabling meaningful similarity and compatibility computation.

---

## Modules

### 1. Similarity Engine (`models/similarity_engine.py`)

Finds players with the most similar profiles using position-weighted latent embeddings and Euclidean distance similarity.

```
Input:  Pedri
Output:
  1. Frenkie de Jong      0.927
  2. Nicolo Barella        0.916
  3. Martin Odegaard       0.915
  4. Warren Zaire-Emery    0.913
  5. Federico Valverde     0.912
```

### 2. Prototype Role Engine (`models/prototype_role_engine.py`)

Classifies players into tactical archetypes using exemplar-based prototype vectors and positional gating.

**Architecture:**
- Layer 1 — Positional gate: restricts eligible roles by FIFA position (e.g. GK cannot be classified as Ball Playing Defender)
- Layer 2 — Weighted prototype matching: cosine similarity on role-specific feature subsets
- Layer 3 — Confidence threshold: returns "Undefined" rather than a wrong label

**Supported roles:**

| Category | Roles |
|---|---|
| Midfield | Deep Playmaker, Creative Playmaker, Ball Winner, Box-to-Box |
| Attack | Wide Winger, Creative Winger, Inside Forward, False 9, Poacher, Target Forward |
| Defense | Ball Playing Defender, Defensive Defender |
| Goalkeeper | Goalkeeper |

```
Input:  Rodri
Output: Deep Playmaker (0.891)
```

### 3. Compatibility Engine (`models/compatibility_engine.py`)

Scores tactical chemistry between two players using a combination of embedding similarity, positional fit, role compatibility, and statistical complementarity.

### 4. Transfer Engine (`models/transfer_engine.py`)

Recommends replacement candidates for a given player filtered by role match, positional fit, and age.

```
Input:  Pedri
Output:
  1. Joao Neves
  2. Gavi
  3. Warren Zaire-Emery
```

### 5. Squad Builder (`models/squad_builder.py`)

Constructs the optimal midfield trio from the dataset by maximizing pairwise chemistry, tactical balance, and role diversity.

```
Output:
  Pedri / Martin Odegaard / Joshua Kimmich
  Tactical Score: 0.946
```

### 6. Starting XI Builder (`models/starting_xi_builder.py`)

Builds a complete 4-3-3 tactical lineup around a selected anchor player using FIFA positions, role bonuses, and compatibility scoring.

### 7. Football Intelligence Report (`models/football_intelligence_report.py`)

Aggregates all modules into a single player report covering role analysis, similar players, compatible partners, transfer candidates, and tactical recommendations.

---

## Visualizations

| Visualization | File | Description |
|---|---|---|
| Latent Space | `visualizations/latent_space_visualization.py` | t-SNE projection of player embeddings colored by position |
| Similarity Chart | `visualizations/similarity_chart.py` | Horizontal bar chart of top similar players |
| Compatibility Heatmap | `visualizations/compatibility_heatmap.py` | Pairwise chemistry matrix |
| Squad Network | `visualizations/squad_network.py` | Graph of player compatibility with role-colored nodes |
| Role Probability Chart | `visualizations/role_probability_chart.py` | Top-5 archetype scores for a player |

All figures are saved to `outputs/` via `save_outputs.py`.

---

## Project Structure

```
ANN_football/
|
|-- footballiq_dashboard.py       # Main entry point
|-- save_outputs.py               # Saves all visualizations to outputs/
|-- requirements.txt
|-- README.md
|
|-- models/
|   |-- autoencoder.py
|   |-- similarity_engine.py
|   |-- compatibility_engine.py
|   |-- prototype_role_engine.py
|   |-- recruitment_assistant.py
|   |-- transfer_engine.py
|   |-- squad_builder.py
|   |-- starting_xi_builder.py
|   |-- football_intelligence_report.py
|   |-- football_intelligence.py
|   |-- chemistry_engine.py
|   |-- latent_embeddings.npy
|   |-- encoder_model.keras
|
|-- visualizations/
|   |-- latent_space_visualization.py
|   |-- similarity_chart.py
|   |-- compatibility_heatmap.py
|   |-- squad_network.py
|   |-- role_probability_chart.py
|
|-- data/
|   |-- final_merged_dataset.csv
|
|-- outputs/
|   |-- latent_space.png
|   |-- similarity_chart.png
|   |-- compatibility_heatmap.png
|   |-- squad_network.png
|   |-- role_probability_chart.png
|
|-- notebooks/
|-- data_preprocessing.py
|-- dataset_merge.py
```

---

## Getting Started

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the dashboard**
```bash
python footballiq_dashboard.py
```

**Save all visualizations**
```bash
python save_outputs.py --player "Pedri"
```

---

## Tech Stack

- **Python** — core language
- **TensorFlow / Keras** — autoencoder training
- **NumPy / Pandas** — data processing
- **Scikit-learn** — similarity metrics, t-SNE
- **Matplotlib / Seaborn** — visualization
- **NetworkX** — squad compatibility graph

---

## License

MIT License

---

## Author

**Rohan Srinivas Ponnana**  
CSE Artificial Intelligence & Machine Learning  
Football Analytics · Representation Learning · Recruitment Intelligence
