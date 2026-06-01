import pandas as pd

stats_df = pd.read_csv("data/players_data-2025_2026.csv")

fifa_df = pd.read_csv("data/player_stats_extra.csv")

stats_df = stats_df.drop_duplicates(subset=["Player"])
stats_df = stats_df.reset_index(drop=True)

fifa_df = fifa_df.drop_duplicates(subset=["Name"])
fifa_df = fifa_df.reset_index(drop=True)

merged_df = pd.merge(
    stats_df,
    fifa_df,
    left_on="Player",
    right_on="Name",
    how="inner"
)

print("Merged Shape:")
print(merged_df.shape)

print("\nSample Players:")
print(
    merged_df[["Player", "Team", "Pace", "Dribbling"]]
    .head()
)

merged_df.to_csv(
    "data/final_merged_dataset.csv",
    index=False
)

print("\nMerged dataset saved successfully!")