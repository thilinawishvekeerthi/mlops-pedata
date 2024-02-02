from datasets import (
    Dataset,
)
from . import append_split_columns_to_dataset
from . import append_index_column_to_dataset

# from . import tag_finder


def preprocessing_pipeline(
    dataset: Dataset,
    add_index: bool = True,
    tag_finder: bool = False,
    add_splits: bool = True,
) -> Dataset:
    """Perform preprocessing on the dataset
    Args
        dataset: dataset to process
        add_index: whether to add an index column to the dataset
        tag_finder: whether to process dataset using tag_finder
        add_splits: whether to add split columns to the dataset
    Returns:
        preprocessed dataset
    Raises:
        TypeError: If the input is not a valid dataset or dictionary of datasets.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Input a valid dataset -> datasets.Dataset - here is {type(dataset)}"
        )

    if add_index:
        # Add index column to dataset
        dataset = append_index_column_to_dataset(dataset)

    if tag_finder:
        # process dataset using tag_finder #TODO - first test tag_finder
        if False:
            dataset = tag_finder(dataset)

    if add_splits:
        # Add split columns to dataset
        dataset = append_split_columns_to_dataset(dataset)

    return dataset
