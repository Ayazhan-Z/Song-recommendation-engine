import pandas as pd
import numpy as np
from numpy.linalg import norm
import streamlit as st

df = pd.read_csv(r'C:\Users\user\Songs Recommender\dataset.csv')

df = df.dropna(subset = ['artists', 'album_name', 'track_name'], axis = 0)
df = df.reset_index(drop = True)
df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first').reset_index(drop=True)

feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

def normalize(data):
    min_val = min(data)
    if min_val<0:
        data = [x+abs(min_val) for x in data]
    max_val = max(data)
    return [x/max_val for x in data]

for i in range(0, len(feature_cols)):
        df[feature_cols[i]] = normalize(df[feature_cols[i]])

def recommend(song, df):
    target = []
    targ = np.array([])

    for i in range(0, df.shape[0]):
        if (song.lower() == df['track_name'][i].lower()):
            target = df.iloc[i]
            for j in range(0, len(feature_cols)):
                targ = np.append(targ, target[feature_cols[j]])
            break

    if (len(target) == 0):
        print("Can't find your song!")
        return
            
    sim = np.array([])
    for i in range(0, df.shape[0]):
        compare = np.array([])
        for j in range(0, len(feature_cols)):
            compare = np.append(compare, df[feature_cols[j]][i])
        temp = np.dot(targ, compare)/(norm(targ) * norm(compare))
        sim = np.append(sim, temp);
    df['sim'] = sim
    
    df = df.sort_values(by = 'sim', ascending = False)
    out = df[1:6]
    return(out[['track_name', 'artists']])


st.header("Songs Recommender Engine!")
st.text("Hello! Want to expand your jam???")

song = st.text_input("Provide Song Name")

if st.button("Search!"):
    st.write("If you like " + song + " listen to:")
    out = (recommend(song, df))
    st.write("### Preview", out.head())



        
    




