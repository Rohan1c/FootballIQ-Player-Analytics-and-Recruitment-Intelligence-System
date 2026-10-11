import sys
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

sys.stdout.reconfigure(
    encoding="utf-8"
)

df = pd.read_csv(
    "data/final_merged_dataset.csv"
)

embeddings = np.load(
    "legacy/models/latent_embeddings.npy"
)

N_CLUSTERS = 8

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    embeddings
)

df["Cluster"] = clusters

print("\nCluster Sizes:\n")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\nSample Players Per Cluster:\n")

for cluster in range(N_CLUSTERS):

    print(f"\n=== Cluster {cluster} ===")

    players = df[
        df["Cluster"] == cluster
    ]["Player"].head(10)

    try:
        print(
            players.to_string(
                index=False
            )
        )

    except Exception:

        for player in players:

            try:
                print(
                    str(player)
                    .encode(
                        "utf-8",
                        errors="ignore"
                    )
                    .decode("utf-8")
                )

            except Exception:
                pass

df.to_csv(
    "data/player_archetypes.csv",
    index=False
)

print(
    "\nplayer_archetypes.csv saved successfully!"
)

pca = PCA(
    n_components=2
)

reduced = pca.fit_transform(
    embeddings
)

plt.figure(
    figsize=(12, 8)
)

scatter = plt.scatter(
    reduced[:, 0],
    reduced[:, 1],
    c=clusters,
    alpha=0.7
)

plt.title(
    "Football Archetype Discovery"
)

plt.xlabel(
    "PCA Component 1"
)

plt.ylabel(
    "PCA Component 2"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.show()