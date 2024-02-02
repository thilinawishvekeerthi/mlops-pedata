import numpy as np
from datasets import (
    Dataset,
)


def append_index_column_to_dataset(dataset: Dataset) -> Dataset:
    """Add an index column to the dataset
    Args:
        dataset: dataset to add index column to

    Returns:
        ds.Dataset: dataset with index column added

    Raises:
        TypeError: If the input type is not datasets.Dataset.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Input a valid dataset -> datasets.Dataset - here is {type(dataset)}"
        )

    if "index" not in dataset.column_names:
        return dataset.add_column("index", np.arange(dataset.num_rows))
    else:
        return dataset
