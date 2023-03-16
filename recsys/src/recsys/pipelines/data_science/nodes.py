from recsys.pipelines.data_science.trainer import CFTrainer


def train_collaborative_filtering_models(
    train_df,
    num_users,
    num_movies,
    num_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
):
    trainer = CFTrainer()
    train_results = trainer.train(
        train_df,
        num_users,
        num_movies,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        test_size=test_size,
        random_state=random_state,
    )

    experiment_data = dict(
        model_type=trainer.model_type,
        best_model=train_results["best_model"],
        figure=train_results["fig"],
    )

    return experiment_data
