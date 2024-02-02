from datasets import (
    Dataset,
)
from ..config import add_encodings
from ..preprocessing import preprocessing_pipeline


def transform_pipeline(
    dataset: Dataset,
    needed_encodings: list[str] = [],
) -> Dataset:
    """Perform base processing on the dataset
    Args
        dataset: dataset to process
        add_index: whether to add an index column to the dataset
        add_splits: whether to add split columns to the dataset
        needed_encoding: encodings to add to the dataset
    Returns:
        ds.Dataset: processed dataset
    Raises:
        TypeError: If the input is not a valid dataset or dictionary of datasets.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Input a valid dataset -> datasets.Dataset - here is {type(dataset)}"
        )

    # preprocessing
    dataset = preprocessing_pipeline(dataset)
    # Add encodings to dataset
    dataset = add_encodings(dataset, needed=needed_encodings)

    return dataset
