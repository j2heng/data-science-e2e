import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

DEFAULT_TEST_SIZE = 0.25


def preprocessing_movies(movies: pd.DataFrame) -> pd.DataFrame:
    movies["genres"] = movies["genres"].str.split("|")

    # Convert the movie genres to binary encoding
    mlb = MultiLabelBinarizer()
    x_genres = movies["genres"]
    one_hot_genres = pd.DataFrame(
        mlb.fit_transform(x_genres), columns=mlb.classes_, index=x_genres.index
    )
    movies = pd.concat([movies, one_hot_genres], axis=1)

    movies = movies.drop(["title", "genres"], axis=1)
    return movies


def preprocessing_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.drop("timestamp", axis=1)

    unique_users = ratings.userId.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)

    unique_movies = ratings.movieId.unique()
    movie_to_index = {old: new for new, old in enumerate(unique_movies)}
    new_movies = ratings.movieId.map(movie_to_index)

    X = pd.DataFrame(
        {
            "userId": new_users,
            "movieId": new_movies,
            "rating": ratings["rating"].astype(np.float32),
        }
    )

    return X, dict(userId_map=user_to_index, movieId_map=movie_to_index)


def get_train_test_data(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    supported_genres: List[str] = None,
    random_state=42,
    shuffle=True,
):
    if supported_genres:
        movie_cols = list(movies.columns)
        supported_cols = set(supported_genres)

        selected_cols = [col for col in movie_cols if col in supported_cols]
        selected_cols.append("movieId")
        movies = movies[selected_cols]

        logging.info(
            f"Genres={list(set(movie_cols).difference(supported_cols))} are not supported."  # noqa:E501
        )

    merged_df = pd.merge(ratings, movies, on="movieId")
    train_df, test_df = train_test_split(
        merged_df, test_size=0.2, random_state=random_state, shuffle=shuffle
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
