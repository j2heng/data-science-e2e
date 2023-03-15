import torch
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, df):
        user_col = "userId"
        movie_col = "movieId"
        rating_col = "rating"
        genres_col = [
            col for col in df.columns if col not in [user_col, movie_col, rating_col]
        ]

        self.users = torch.tensor(df[user_col].values, dtype=torch.long)
        self.movies = torch.tensor(df[movie_col].values, dtype=torch.long)
        self.genres = torch.tensor(df[genres_col].values, dtype=torch.long)
        self.ratings = torch.tensor(df[rating_col].values, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        movie = self.movies[idx]
        genre = self.genres[idx]
        rating = self.ratings[idx]
        return user, movie, genre, rating
