import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("data/player_archetypes.csv")

important_features = [
    "Pace",
    "Shooting",
    "Passing",
    "Dribbling",
    "Defending",
    "Physicality"
]

results = []

print("\nTACTICAL ARCHETYPE ANALYSIS\n")

for cluster in sorted(df["Cluster"].unique()):

    cluster_df = df[df["Cluster"] == cluster]

    print("\n" + "=" * 50)
    print(f"CLUSTER {cluster}")
    print("=" * 50)

    print(f"\nPlayers: {len(cluster_df)}")

    averages = (
        cluster_df[important_features]
        .mean()
        .round(2)
    )

    print("\nAverage Attributes:\n")
    print(averages)

    results.append({
        "Cluster": cluster,
        "Players": len(cluster_df),
        "Pace": averages["Pace"],
        "Shooting": averages["Shooting"],
        "Passing": averages["Passing"],
        "Dribbling": averages["Dribbling"],
        "Defending": averages["Defending"],
        "Physicality": averages["Physicality"]
    })

summary_df = pd.DataFrame(results)

summary_df.to_csv(
    "data/archetype_summary.csv",
    index=False
)

print(
    "\nArchetype summary saved to:"
)
print(
    "data/archetype_summary.csv"
)