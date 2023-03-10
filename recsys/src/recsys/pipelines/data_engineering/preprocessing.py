from typing import List
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import logging

DEFAULT_TEST_SIZE = 0.25

def preprocessing_movies(movies: pd.DataFrame) -> pd.DataFrame:
    movies["genres"] = movies["genres"].str.split("|")

    # Convert the movie genres to binary encoding
    mlb = MultiLabelBinarizer()
    x_genres = movies["genres"]
    one_hot_genres = pd.DataFrame(mlb.fit_transform(x_genres), columns=mlb.classes_,index=x_genres.index)
    movies = pd.concat([movies, one_hot_genres], axis=1)
    
    movies = movies.drop(["title", "genres"], axis=1)
    return movies


def preprocessing_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.drop("timestamp", axis=1)
    return ratings


def get_train_test_data(movies: pd.DataFrame, ratings: pd.DataFrame, supported_genres: List[str] = None, random_state=42, shuffle=True):
    movies = preprocessing_movies(movies)
    ratings = preprocessing_ratings(ratings)

    if supported_genres:
        movie_cols = list(movies.columns)
        supported_cols = set(supported_genres)

        selected_cols = [col for col in movie_cols if col in supported_cols]
        selected_cols.append("movieId")
        movies = movies[selected_cols]

        logging.info(f"Genres={list(set(movie_cols).difference(supported_cols))} are not supported.")

    merged_df = pd.merge(ratings, movies, on='movieId')
    train_df, test_df = train_test_split(merged_df, test_size=0.2)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
