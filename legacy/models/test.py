from legacy.models.prototype_role_engine import get_primary_role
import pandas as pd
import sys

# Handle Unicode encoding on Windows
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("data/final_merged_dataset.csv")
matches = df[
    df["Player"]
    .str.contains(
        "Ode",
        case=False,
        na=False
    )
]

print(
    matches["Player"]
    .tolist()[:50]
)