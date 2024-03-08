import streamlit as st
import pandas as pd
import nltk
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data and similarity matrix
df = pd.read_csv("/Music Recommendation System/MusicRecommendationSystem/newsongs.csv")

# Tokenization and stemming
nltk.download('punkt')

ps = PorterStemmer()

def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = (ps.stem(w) for w in tokens)
    return " ".join(stemming)

df['text'] = df['text'].str.lower().replace(r'[^\w\s]', '').replace(r'\n', '', regex=True)
df['text'] = df['text'].apply(lambda x: tokenization(x))

# TF-IDF Vectorization
tfid = TfidfVectorizer(stop_words="english")
matrix = tfid.fit_transform(df['text'])

# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(matrix)

def recommendation(song):
    idx = df[df['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    songs = []
    for i in distances[1:21]:
        songs.append(df.iloc[i[0]])
    return songs

st.title('Music Recommendation')

def get_user_input():
    user_input = {}
    for field in df.columns.drop(["artist", "link", "text"]):
        user_input[field] = st.text_input(f"Enter {field}")
    return user_input

def main():
    user_input = get_user_input()
    if st.button('recommend'):
        recommends = recommendation(user_input['song'])
        st.write("Recommended Songs:")
        for song in recommends:
            st.write(f"**{song['song']}**")
            # st.page_link(song['link'], width=200)  # Display image
            st.write(f"Artist: {song['artist']}")
            st.write(f"Link: {song['link']}")

if __name__ == '__main__':
    main()
