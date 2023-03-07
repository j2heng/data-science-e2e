import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from recsys.pipelines.data_science.torch_dataset import MovieDataset

def generate_sample_data_frame(num_samples):

    # Create a sample dataframe with random data
    data = pd.DataFrame({'userId': np.random.randint(1, 11, num_samples),
                        'movieId': np.random.randint(1, 21, num_samples),
                        'rating': np.random.randint(1, 6, num_samples),
                        'Action': np.random.randint(0, 1, num_samples),
                        'Comedy': np.random.randint(0, 1, num_samples),
    })
    return data



class TestWrapper:
    def setup(self):
        self.data = generate_sample_data_frame(20)

    def test_dataloader(self):
        dataloader = DataLoader(MovieDataset(self.data), batch_size=5, shuffle=False)
        assert len(dataloader) == 4

        for _, batch in enumerate(dataloader):
            user_input, movie_input, genres_input, ratings = batch

            assert isinstance(user_input, torch.Tensor)
            assert isinstance(movie_input, torch.Tensor)
            assert isinstance(genres_input, torch.Tensor)
            assert isinstance(ratings, torch.Tensor)

            assert user_input.shape == (5,)
            assert movie_input.shape == (5,)
            assert genres_input.shape ==(5, 2)
            assert ratings.shape == (5,)
            