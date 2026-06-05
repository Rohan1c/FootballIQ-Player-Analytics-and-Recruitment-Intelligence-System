import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("data/final_merged_dataset.csv")

selected_features = [

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

    "Gls",
    "Ast",
    "G+A",
    "G-PK",
    "PK",
    "PKatt",

    "Sh",
    "SoT",
    "Sh/90",
    "SoT/90",
    "G/Sh",

    "Crs",
    "Fld",
    "Fls",

    "+/-",
    "+/-90",

    "TklW",
    "Int",

    "CrdY",

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

df = df.drop_duplicates(subset=["Player"])

for col in selected_features:

    if col in df.columns:

        if df[col].dtype != "object":

            df[col] = df[col].fillna(
                df[col].median()
            )

df = df[
    ["Player", "Pos"] +
    selected_features
]

df = df.reset_index(drop=True)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    df[selected_features]
)

input_dim = scaled_data.shape[1]

input_layer = Input(
    shape=(input_dim,)
)

encoded = Dense(
    256,
    activation="relu"
)(input_layer)

encoded = Dropout(
    0.20
)(encoded)

encoded = Dense(
    128,
    activation="relu"
)(encoded)

encoded = Dropout(
    0.20
)(encoded)

encoded = Dense(
    64,
    activation="relu",
    name="latent_space"
)(encoded)

decoded = Dense(
    128,
    activation="relu"
)(encoded)

decoded = Dense(
    256,
    activation="relu"
)(decoded)

decoded = Dense(
    input_dim,
    activation="linear"
)(decoded)

autoencoder = Model(
    input_layer,
    decoded
)

encoder = Model(
    input_layer,
    encoded
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

history = autoencoder.fit(
    scaled_data,
    scaled_data,
    epochs=200,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)

latent_embeddings = encoder.predict(
    scaled_data
)

print("\nEmbedding Shape:")
print(latent_embeddings.shape)

np.save(
    "models/latent_embeddings.npy",
    latent_embeddings
)

encoder.save(
    "models/encoder_model.keras"
)

print("\nEmbeddings saved successfully!")
print("Encoder model saved!")