import sys
import os

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from legacy.models.compatibility_engine import (
    compatibility_score,
    find_player,
    df
)

players = [

    "Pedri",
    "Rodri",
    "Valverde",
    "Bellingham",
    "Haaland"

]

valid_players = []
valid_indices = []

for player in players:

    idx = find_player(player)

    if idx is not None:

        valid_players.append(
            df.iloc[idx]["Player"]
        )

        valid_indices.append(
            idx
        )

n = len(valid_indices)

matrix = np.zeros(
    (n,n)
)

for i in range(n):

    for j in range(n):

        matrix[i,j] = compatibility_score(

            valid_indices[i],

            valid_indices[j]

        )

plt.figure(
    figsize=(12,9)
)

sns.heatmap(

    matrix,

    annot=True,

    fmt=".2f",

    cmap="RdYlGn",

    linewidths=1,

    square=True,

    xticklabels=valid_players,

    yticklabels=valid_players,

    cbar_kws={

        "label":
        "Compatibility Score"

    }

)

plt.title(

    "FootballIQ Player Compatibility Matrix",

    fontsize=18,

    fontweight="bold"

)

plt.tight_layout()

plt.show()