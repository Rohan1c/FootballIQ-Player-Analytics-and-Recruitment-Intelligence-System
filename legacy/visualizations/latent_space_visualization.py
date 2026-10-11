import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

df = pd.read_csv(
    "data/final_merged_dataset.csv"
)

embeddings = np.load(
    "legacy/models/latent_embeddings.npy"
)

df = df.reset_index(drop=True)

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

emb_2d = tsne.fit_transform(
    embeddings
)

position_colors = {

    "CB":"red",
    "RB":"red",
    "LB":"red",

    "CDM":"green",
    "CM":"green",
    "CAM":"green",

    "RM":"blue",
    "LM":"blue",
    "RW":"blue",
    "LW":"blue",
    "ST":"blue",

    "GK":"black"
}

colors = [

    position_colors.get(
        pos,
        "gray"
    )

    for pos in df["Position"]

]

plt.figure(
    figsize=(15,10)
)

plt.scatter(

    emb_2d[:,0],

    emb_2d[:,1],

    c=colors,

    alpha=0.6,

    s=35

)

important_players = [

    "Pedri",
    "Rodri",
    "Valverde",
    "Bellingham",
    "Haaland",
    "Palmer",
    "Vinicius"

]

for player in important_players:

    rows = df[
        df["Player"]
        .str.contains(
            player,
            case=False,
            na=False
        )
    ]

    if len(rows):

        idx = rows.index[0]

        plt.annotate(

            rows.iloc[0]["Player"],

            (
                emb_2d[idx,0],
                emb_2d[idx,1]
            ),

            fontsize=10,

            fontweight="bold"

        )

plt.title(

    "FootballIQ Latent Embedding Space",

    fontsize=18,

    fontweight="bold"

)

plt.xlabel("TSNE Dimension 1")
plt.ylabel("TSNE Dimension 2")

plt.grid(alpha=0.2)

plt.tight_layout()

plt.show()