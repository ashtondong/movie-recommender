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
# This genre index will be used as part of the genre matrix
genre_index = {g:i for i,g in enumerate(all_genres)}
# Check that the genre_index is correct
    # print(genre_index)



