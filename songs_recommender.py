import pandas as pd
import numpy as np
from numpy.linalg import norm
import streamlit as st
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "dataset.csv"))

df = df.dropna(subset = ['artists', 'album_name', 'track_name'], axis = 0)
df = df.reset_index(drop = True)
df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first').reset_index(drop=True)

feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

def recommend(song, df):
    matches = df[df['track_name'].str.lower() == song.lower()]
    if matches.empty:
        return None

    target_vector = matches.iloc[0][feature_cols].values.reshape(1, -1)
    all_vectors = df[feature_cols].values

    sim_scores = cosine_similarity(target_vector, all_vectors)[0]
    df = df.copy()
    df['sim'] = sim_scores

    df_sorted = df.sort_values(by='sim', ascending=False)
    out = df_sorted[1:6]
    return out[['track_name', 'artists']]

st.header("Songs Recommender Engine!")
st.text("Hello! Want to expand your jam???")

song = st.text_input("Provide Song Name")

if st.button("Search!"):
    out = (recommend(song, df))
    if out is not None:
        st.write("If you like " + song + " listen to:")
        st.write("### Recommendations", out)
    else:
        st.write("Can't find that song in the dataset. Try another title!")



        
    




