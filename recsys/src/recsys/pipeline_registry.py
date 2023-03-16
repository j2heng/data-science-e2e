"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from recsys.pipelines.data_engineering import pipeline as de
from recsys.pipelines.data_science import pipeline as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    pipelines = {}
    pipelines["preprocessing_dataset"] = de.preprocessing_dataset_pipeline()
    pipelines["train_cf_model"] = ds.train_cf_model_pipeline()

    return pipelines
