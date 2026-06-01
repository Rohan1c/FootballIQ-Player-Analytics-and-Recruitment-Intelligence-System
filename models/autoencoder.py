import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

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

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df[selected_features])

input_dim = scaled_data.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(128, activation="relu")(input_layer)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(decoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(input_layer, decoded)

encoder = Model(input_layer, encoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.fit(
    scaled_data,
    scaled_data,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

latent_embeddings = encoder.predict(scaled_data)

print(latent_embeddings[:5])

np.save(
    "models/latent_embeddings.npy",
    latent_embeddings
)

encoder.save("models/encoder_model.keras")

print("Embeddings saved successfully!")
print("Encoder model saved!")