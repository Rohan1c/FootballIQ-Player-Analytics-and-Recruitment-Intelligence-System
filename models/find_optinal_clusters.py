import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

embeddings = np.load(
    "models/latent_embeddings.npy"
)

best_k = None
best_score = -1

print("\nFinding Optimal Number of Clusters...\n")

for k in range(5, 21):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        embeddings
    )

    score = silhouette_score(
        embeddings,
        labels
    )

    print(
        f"K = {k} | Silhouette Score = {score:.4f}"
    )

    if score > best_score:

        best_score = score
        best_k = k

print("\n" + "=" * 50)

print(
    f"Best K: {best_k}"
)

print(
    f"Best Score: {best_score:.4f}"
)

print("=" * 50)