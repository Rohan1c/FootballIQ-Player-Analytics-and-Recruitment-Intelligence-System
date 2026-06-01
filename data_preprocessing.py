import pandas as pd

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/final_merged_dataset.csv")

selected_features = [

    # Usage / tactical importance
    "Age_x",
    "MP",
    "Starts",
    "Min",
    "Mn/Start",
    "Mn/Sub",
    "Min%",
    "Compl",
    "PPM",
    "On-Off",

    # Match production
    "Gls",
    "Ast",
    "G+A",
    "G-PK",
    "PK",
    "PKatt",

    # Shooting
    "Sh",
    "SoT",
    "Sh/90",
    "SoT/90",
    "G/Sh",

    # Creativity / progression
    "Crs",
    "Fld",
    "Fls",

    # Tactical impact
    "+/-",
    "+/-90",

    # Defensive
    "TklW",
    "Int",

    # Discipline
    "CrdY",

    # FIFA STYLE ATTRIBUTES 😭🔥

    # Pace / movement
    "Pace",
    "Acceleration",
    "Sprint Speed",

    # Shooting style
    "Shooting",
    "Positioning",
    "Finishing",
    "Shot Power",
    "Long Shots",
    "Volleys",
    "Penalties",

    # Creativity / passing
    "Passing",
    "Vision",
    "Crossing",
    "Free Kick Accuracy",
    "Short Passing",
    "Long Passing",
    "Curve",

    # Dribbling / agility
    "Dribbling",
    "Agility",
    "Balance",
    "Reactions",
    "Ball Control",
    "Composure",

    # Defensive IQ
    "Defending",
    "Interceptions",
    "Heading Accuracy",
    "Def Awareness",
    "Standing Tackle",
    "Sliding Tackle",

    # Physical profile
    "Physicality",
    "Jumping",
    "Stamina",
    "Strength",
    "Aggression"
]

df = df.drop_duplicates(subset=["Player"])

for col in selected_features:

    if df[col].dtype != "object":

        df[col] = df[col].fillna(df[col].median())

df = df[["Player", "Pos"] + selected_features]

df = df.reset_index(drop=True)

# Reduce FIFA influence 😭🔥
fifa_features = [
    "Pace",
    "Acceleration",
    "Sprint Speed",
    "Shooting",
    "Positioning",
    "Finishing",
    "Shot Power",
    "Long Shots",
    "Volleys",
    "Penalties",
    "Passing",
    "Vision",
    "Crossing",
    "Free Kick Accuracy",
    "Short Passing",
    "Long Passing",
    "Curve",
    "Dribbling",
    "Agility",
    "Balance",
    "Reactions",
    "Ball Control",
    "Composure",
    "Defending",
    "Interceptions",
    "Heading Accuracy",
    "Def Awareness",
    "Standing Tackle",
    "Sliding Tackle",
    "Physicality",
    "Jumping",
    "Stamina",
    "Strength",
    "Aggression"
]

df[fifa_features] = df[fifa_features] * 0.3

print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df[selected_features])

print("\nScaled Data Sample:\n")
print(scaled_data[:5])