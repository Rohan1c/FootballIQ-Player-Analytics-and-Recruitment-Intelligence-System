import pandas as pd

df = pd.read_csv("data/players_data_light.csv")

selected_features = [
    "Player",
    "Pos",
    "Age",
    "MP",
    "Starts",
    "Min",
    "Gls",
    "Ast",
    "G+A",
    "CrdY",
    "Sh",
    "SoT",
    "Sh/90",
    "SoT/90",
    "G/Sh",
    "Crs",
    "TklW",
    "Int",
    "Fld"
]

df = df[selected_features]

print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())

df["G/Sh"] = df["G/Sh"].fillna(0)

print("\nRemaining Missing Values:")
print(df.isnull().sum())