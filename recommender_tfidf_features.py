import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

if os.path.exists("movies.pkl"):
    print("Loading csv...\n")
    joblib.load("movies.pkl")

else:

    print("Reading csv...\n")
    # Read in csv file
    movies = pd.read_csv("dataset/TMDB_movie_dataset_v11.csv", encoding="utf-8")
    # Checking fields:
        # print(movies.head())

    # Make sure all empty genre and keyword fields have a string associated with it
    movies["keywords"] = movies["keywords"].fillna("")
    movies["genres"] = movies["genres"].fillna("")
    movies["overview"] = movies["overview"].fillna("")
    movies["adult"] = movies["adult"].fillna("")
    movies["adult"] = movies["adult"].astype(bool)

    # id -> index
    indices = pd.Series(movies.index, index=movies["id"])

    # Define features to be a combination of keyword, overview, genre
    movies["features"] = (movies["genres"] + movies["keywords"] + movies["overview"])

    joblib.dump(movies, "movies.pkl")


# Optimize load time by creating and saving the tfidf objects
if os.path.exists("tfidf.pkl") and os.path.exists("tfidf_matrix.pkl"):
    print("Loading saved TF-IDF...\n")
    tfidf = joblib.load("tfidf.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
else:
    print("Building TF-IDF...\n")
    # This creates the empty tfidf object
    tfidf = TfidfVectorizer()

    # The fit learns the vocab and the tranform puts it into a vector form with movies as row, features as column
    # and tfidf scores as values
    tfidf_matrix = tfidf.fit_transform(movies["features"])

    # Save the objects
    joblib.dump(tfidf, "tfidf.pkl")
    joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")


# Create the recommend function
def recommend(title):
    # Find the closest matching title
    # This works by turning movies["title"] into a boolean mask based on if it is closely matching title input
    # Case = False makes it case insensitive
    # na = False treats empty cells as False
    # The outer movies[...] only returns the titles that returned True from the internal filter
    matches = movies[movies["title"].str.contains(title, case=False, na=False)]
    
    if len(matches) == 0:
        return "No movie found!\n"
    
    # Trim matches
    matches = matches.head(5)
    
    # Print the whole match list so user can select exact movie
    print("Found: \n")
    # i is for user selection, idx is the actual movie index, row is the whole row of dataset, select title from row
    for i, (idx,row) in enumerate(matches.iterrows()):
        print(f"{i}: {row['title']}")

    choice = int(input("Select a number: "))

    # Movie ID of the selected
    selected = matches.iloc[choice]["id"]

    # index of the selected movie
    # Need this for the tfidf matrix
    idx = indices[selected]

    is_adult = movies.iloc[idx]["adult"]

    # DEBUG
    # print(is_adult)

    # Build a candidate set to prevent inappropriate results
    # This pulls the movie indicies of child-friendly movies
    # Candidate_indicies contain original movie indicies
    if not is_adult:
        candidate_indices = movies[movies["adult"] == False].index
    else:
        candidate_indices = movies.index

        # DEBUG
        # print("candidate size: ", len(candidate_indices))

    # Instead of comparing all movies to all movies for the score, compare just the requested movie across all appropriate candidates
    # This optimizes run time
    # The scores matrix
    # cosine_similarity compares one vector tfidf_matrix[idx] to all feature vectors for all movies
    # The output values are the similarity scores
    # Need to flatten to enumerate
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix[candidate_indices]).flatten()


    # Get similarity scores for the movie
    # Output [(idx, score),...]
    similarity_scores = list(enumerate(scores))

    # Sort similarity scores by score, index 1, from highest to lowest
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)


    # Remove itself (the first result is always the movie itself) since it'll always be 1.0
    # Splice to trim recommended movies down to top 5
    similarity_scores = similarity_scores[1:6]

    # i in similarity_scores produces (candidate index, score)
    # Must map back from candidate indices to movie indices
    mapping_indices = [i[0] for i in similarity_scores]
    movie_indices = candidate_indices[mapping_indices]

    # iloc finds movies by integer position and returns a subset of pandas Series movies["titles"]
    # iloc is not a method but a special indexer in pandas
    # Convert the pandas Series to a list output
    return movies["title"].iloc[movie_indices].tolist()

# This is for direct run in CML
if __name__ == "__main__":
    searching = True
    cache = {}
    while searching:
        title = input("Enter a movie title: ")
        if title in cache:
            print(cache[title])
        elif title == "..":
            searching = False
        else: 
            recommendations = recommend(title)
            cache[title] = recommendations
            print(recommendations)