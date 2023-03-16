from kedro.pipeline import Pipeline, node

from recsys.pipelines.data_science.nodes import train_collaborative_filtering_models


def train_cf_model_pipeline():
    return Pipeline(
        [
            node(
                func=train_collaborative_filtering_models,
                inputs=dict(
                    train_df="train_df",
                    num_users="params:num_users",
                    num_movies="params:num_movies",
                    num_epochs="params:num_epochs",
                    batch_size="params:batch_size",
                ),
                outputs="experiment_data_cf",
                name="train_collaborative_filtering_models",
            )
        ]
    )
