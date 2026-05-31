
# models/autoencoder.py

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("data/players_data_light.csv")

selected_features = [
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

df["Age"] = df["Age"].fillna(df["Age"].median())

df["G/Sh"] = df["G/Sh"].fillna(0)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df[selected_features])

input_dim = scaled_data.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(24, activation="relu")(input_layer)

encoded = Dense(16, activation="relu")(encoded)

latent = Dense(8, activation="relu")(encoded)

decoded = Dense(16, activation="relu")(latent)

decoded = Dense(24, activation="relu")(decoded)

output_layer = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

autoencoder.fit(
    scaled_data,
    scaled_data,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stopping]
)

encoder = Model(
    inputs=input_layer,
    outputs=latent
)

latent_embeddings = encoder.predict(scaled_data)

np.save(
    "models/latent_embeddings.npy",
    latent_embeddings
)

encoder.save(
    "models/encoder_model.keras"
)

print("\nLatent Embeddings:\n")

print(latent_embeddings[:5])

print("\nEmbeddings saved successfully!")

print("Encoder model saved!")

