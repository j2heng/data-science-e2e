from kedro.pipeline import Pipeline, node
from recsys.pipelines.data_engineering.preprocessing import get_train_test_data

def preprocessing_dataset_pipeline():
    return Pipeline(
        [
            node(
                func=get_train_test_data,
                inputs=dict(
                    movies="raw_movies",
                    ratings="raw_ratings",
                    supported_genres="params:supported_genres"
                ),
                outputs=["train_df", "test_df"],
                name="get_train_test_data"
            ), 
        ]
    )
