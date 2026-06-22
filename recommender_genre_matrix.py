import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Read in csv file
movies = pd.read_csv("movielens_100k.csv", encoding="utf-8")
# Checking fields:
    # print(movies.head())

# Make sure all empty genre fields have a string associated with it
movies["genres"] = movies["genres"].fillna("")

# Construct a genre list to add all genres into a set in prep for the genre_matrix
genre_list = movies["genres"].str.split()
# Checking fields:
    # print(genre_list.head())

# Create a genre index for the matrix
# The set removes all duplicates
all_genres = set()

# Genre_list is a double array
for genres in genre_list:
    # Grab the genre
    for genre in genres:
        # Add the genre to all_genres if not duplicate
        all_genres.add(genre)
all_genres = sorted(all_genres)

# This genre index will be used to form the genre matrix
genre_index = {g:i for i,g in enumerate(all_genres)}
# Check that the genre_index is correct
    # print(genre_index)

# Create genre_matrix using Numpy
# This "matrix" is formed from genre vectors
genre_matrix = np.zeros((len(movies), len(all_genres)))
for i, genres in enumerate(genre_list):
    # Mark the genre for a movie
    for genre in genres:
        genre_matrix[i, genre_index[genre]] = 1
# Check that genre_matrix is correct
    # print(genre_matrix)

# Use cosinesimilarity to create a 2D numpy array that compares each movie to itself and all other movies
# The each number in the array is the similarity number between movie i and movie j
# Cosine_similarity = (A dot B) / (magnitude(A)*magnitude(B))
# Note similarity[i][j] = similarity[j][i]
similarity = cosine_similarity(genre_matrix)
# Check that the similarity matrix is correct
    # print(similarity)

# Create an index/title column to map titles to indices
indices = pd.Series(movies.index, index=movies["title"])
    # print(indices)

# Create the recommend function
def recommend(title, similarity=similarity):
    # Find the closest matching title
    # This works by turning movies["title"] into a boolean mask based on if it is closely matching title input
    # Case = False makes it case insensitive
    # na = False treats empty cells as False
    # The outer movies[...] only returns the titles that returned True from the internal filter
    matches = movies[movies["title"].str.contains(title, case=False, na=False)]
    
    if len(matches) == 0:
        return "No movie found!"
    
    # Get index of movie, first in the list of matches
    idx = matches.index[0]

    # Get similarity scores for the movie
    similarity_scores = list(enumerate(similarity[idx]))

    # Sort similarity scores by score, index 1, from highest to lowest
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Remove itself (the first result is always the movie itself) since it'll always be 1.0
    # Splice to trim recommended movies down to top 5
    similarity_scores = similarity_scores[1:6]

    # i in similarity_scores produces (movie index, score)
    movie_indices = [i[0] for i in similarity_scores]

    # iloc finds movies by integer position and returns a subset of pandas Series movies["titles"]
    # iloc is not a method but a special indexer in pandas
    return movies["title"].iloc[movie_indices]

# This is for direct run in CML
if __name__ == "__main__":
    searching = True
    while searching:
        title = input("Enter a movie title: ")
        if title == "..":
            searching = False
        else: 
            print(recommend(title))


