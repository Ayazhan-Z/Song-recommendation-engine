## **🎵 Song Recommendation Engine**
A content-based song recommender built with Python, Pandas, and Streamlit. Enter a song you like, and it suggests similar tracks based on audio features like danceability, energy, and tempo.

### **How It Works**
1) Loads and cleans Spotify track data
2) Normalizes 9 audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo)
3) Computes cosine similarity between your chosen song and every other track
4) Returns the top 5 most similar songs

### **Dataset**
Spotify Tracks Dataset by maharshipandya (Kaggle) — ~114,000 tracks, 125 genres.
Download dataset.csv and place it in a data/ folder before running.

### **Setup**
```bash 
bashgit clone https://github.com/Ayazhan-Z/Song-recommendation-engine.git
cd Song-recommendation-engine
pip install -r requirements.txt
streamlit run songs_recommender.py
```

### **Tech Stack**
Python · Pandas · NumPy · Streamlit

### **Future Improvements**
- Vectorize similarity computation for speed
- Use sklearn.preprocessing.MinMaxScaler for normalization
- Handle duplicate song titles by different artists
- Add genre/mood filtering
Author

Ayazhan — @Ayazhan-Z
