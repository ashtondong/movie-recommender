import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Read in csv file
movies = pd.read_csv("movielens_100k.csv", encoding="utf-8")
# Checking fields:
    # print(movies.head())

# Make sure all empty genre fields have a string associated with it
movies["genres"] = movies["genres"].fillna("")

# Construct a genre list
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


