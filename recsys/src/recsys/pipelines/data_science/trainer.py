import logging

import plotly.graph_objs as go
import torch
import torch.nn as nn
import torch.optim as optim
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from recsys.pipelines.data_science.core_model import (
    CollaborativeFiltering,
    MatrixFactorization,
)
from recsys.pipelines.data_science.torch_dataset import MovieDataset


class BaseModelTrainer:
    model_type: str

    def __init__(self, core_model, loss_fn) -> None:
        self.model = core_model
        self.loss_fn = loss_fn

        logging.info(f"model={self.model}")
        logging.info(f"loss_fn={self.loss_fn}")

    def train(
        self,
        train_df,
        num_users,
        num_movies,
        num_epochs=100,
        batch_size=64,
        lr=0.001,
        patience=10,
        test_size=0.2,
        random_state=42,
    ):
        # Define the model with embedding layers for userId and movieId
        model = self.model(num_users, num_movies)

        # Get minmax rating
        minmax = train_df.rating.min().astype(float), train_df.rating.max().astype(
            float
        )

        # Define the loss function and optimizer
        criterion = self.loss_fn
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Define the train and test data loaders
        train, val = train_test_split(
            train_df, test_size=test_size, random_state=random_state
        )

        train_loader = DataLoader(
            MovieDataset(train), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(MovieDataset(val), batch_size=batch_size, shuffle=False)

        # Set up early stopping parameters
        best_val_loss = float("inf")
        best_model = None
        epoch_counter = 0

        # Define lists to store the losses
        train_losses = []
        val_losses = []

        # Train the model on the training set
        for epoch in range(num_epochs):
            train_loss = 0.0
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                # Train the model on the current batch
                user_input, movie_input, genres_input, ratings = batch
                optimizer.zero_grad()
                outputs = model(user_input, movie_input, minmax)
                loss = criterion(outputs, ratings)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Evaluate the model on the validation set
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    user_input, movie_input, genres_input, ratings = batch
                    # TODO: add minmax transform to ratings
                    outputs = model(user_input, movie_input, minmax)
                    loss = criterion(outputs, ratings)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # Add the epoch losses to the lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            logging.info(
                f"[{(epoch+1):03d}/{num_epochs:03d}] train: {train_loss:.4f} - val: {val_loss:.4f}"  # noqa:E501
            )

            # Check if the validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                epoch_counter = 0
                # Save the model parameters
                torch.save(best_model, "best_model.pkl")
            else:
                epoch_counter += 1
                if epoch_counter >= patience:
                    logging.info(
                        "Validation loss has not improved for {} epochs. Stopping training.".format(  # noqa:E501
                            epoch_counter
                        )
                    )
                    break

        # Plot train/validation loss
        fig = self._plot_train_val_loss(train_losses, val_losses)

        return dict(best_model=best_model, fig=fig)

    def _plot_train_val_loss(self, train_losses, val_losses):
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Train/Validation Loss"))

        fig.add_trace(go.Scatter(y=train_losses, mode="lines", name="Train Loss"))
        fig.add_trace(go.Scatter(y=val_losses, mode="lines", name="Validation Loss"))

        fig.update_layout(
            title="Train/Validation Loss", xaxis_title="Epoch", yaxis_title="Loss"
        )

        # Save the plot as an HTML file
        fig.write_html("loss_plot.html")
        return fig


class CFTrainer(BaseModelTrainer):
    model_type = "CollaborativeFiltering (userId, movieId)"

    def __init__(self):
        super().__init__(core_model=CollaborativeFiltering, loss_fn=nn.MSELoss())


class MFTrainer(BaseModelTrainer):
    model_type = "MatrixFactorization (userId, movieId)"

    def __init__(self):
        super().__init__(core_model=MatrixFactorization, loss_fn=nn.MSELoss())
