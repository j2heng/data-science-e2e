import torch
import torch.nn as nn


class CollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_movies, emb_size=512):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_size)
        self.item_emb = nn.Embedding(n_movies, emb_size)
        self.fc1 = nn.Linear(emb_size * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user_ids, item_ids, minmax=None):
        user_embedding = self.user_emb(user_ids)
        item_embedding = self.item_emb(item_ids)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.Sigmoid()(x)
        if minmax is not None:
            min_rating, max_rating = minmax
            x = x * (max_rating - min_rating + 1) + min_rating - 0.5
        return x

    def predict(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, emb_size=512):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_size, sparse=True)
        self.item_emb = nn.Embedding(n_movies, emb_size, sparse=True)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_movies, 1)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_emb(user_ids)
        item_embedding = self.item_emb(item_ids)
        user_bias = self.user_biases(user_ids)
        item_bias = self.item_biases(item_ids)
        dot_product = torch.sum(torch.mul(user_embedding, item_embedding), dim=1)
        output = dot_product + user_bias.squeeze() + item_bias.squeeze()

        return output.squeeze()

    def predict(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)
