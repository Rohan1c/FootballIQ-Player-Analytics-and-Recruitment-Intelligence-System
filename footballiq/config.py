"""Paths and constants shared by every module (independent of the working directory)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "evaluation" / "results"
OUTPUT_DIR = ROOT / "outputs"

FBREF_RAW = DATA_DIR / "players_data-2025_2026.csv"
EA_RAW = DATA_DIR / "player_stats_extra.csv"
PLAYERS_CSV = DATA_DIR / "players_linked.csv"          # output of footballiq.linkage
LEGACY_MERGED = DATA_DIR / "final_merged_dataset.csv"  # original exact-name merge (baseline)
LEGACY_DIR = ROOT / "legacy"
LEGACY_EMBEDDINGS = LEGACY_DIR / "models" / "latent_embeddings.npy"
CORPUS_DIR = DATA_DIR / "corpus"                       # multi-season corpus, see scripts/build_corpus.py
CORPUS_SEASONS = CORPUS_DIR / "player_seasons.csv.gz"
CORPUS_TRANSFERS = CORPUS_DIR / "transfers.csv.gz"
ROLE_LABELS = ROOT / "evaluation" / "role_labels.csv"                        # the project's own labels
ROLE_LABELS_DRAFT = ROOT / "evaluation" / "role_labels_draft_unverified.csv"  # see evaluation/README.md

SEED = 0
MIN_MINUTES = 450          # players below this are embedded but flagged as low-sample
EMBEDDING_METHOD = "pca"   # "pca" (default) or "mdae" - see footballiq/embedding.py
EMBEDDING_DIM = 16
