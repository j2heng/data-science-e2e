from kedro.pipeline import Pipeline, node
from recsys.pipelines.data_engineering.preprocessing import (
    get_train_test_data,
    preprocessing_movies,
    preprocessing_ratings,
)


def preprocessing_dataset_pipeline():
    return Pipeline(
        [
            node(
                func=preprocessing_movies,
                inputs=dict(
                    movies="raw_movies",
                ),
                outputs="processed_movies",
                name="preprocessing_movies",
            ),
            node(
                func=preprocessing_ratings,
                inputs=dict(
                    ratings="raw_ratings",
                ),
                outputs=["processed_ratings", "ids_mapping"],
                name="preprocessing_ratings",
            ),
            node(
                func=get_train_test_data,
                inputs=dict(
                    movies="processed_movies",
                    ratings="processed_ratings",
                    supported_genres="params:supported_genres",
                ),
                outputs=["train_df", "test_df"],
                name="get_train_test_data",
            ),
        ]
    )
