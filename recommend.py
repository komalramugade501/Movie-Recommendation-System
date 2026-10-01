import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/movies.csv")
df["features"] = (
    df["genre"].fillna("") + " " + df["keywords"].fillna("")
)

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(df["features"])
similarity = cosine_similarity(matrix)

indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

def recommend(movie_name, n=5):
    key = movie_name.lower()

    if key not in indices:
        raise ValueError(f"Movie not found: {movie_name}")

    idx = indices[key]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)

    recommendations = [
        df.iloc[i]["title"]
        for i, score in scores[1:n + 1]
    ]
    return recommendations

movie_name = "Inception"
print(f"Recommendations for: {movie_name}")
for movie in recommend(movie_name):
    print("-", movie)
