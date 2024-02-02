"""
This file contains the abstract base class for splitters.
DatasetSplitter

it contains the following splitter classes:
- DatasetSplitterRandomTrainTest class for random train test split

"""

from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict, concatenate_datasets
from itertools import combinations


class DatasetSplitter(ABC):
    """Dataset Splitter for random Train Test Split."""

    def __init__(
        self,
        seed: int = 42,
        append_split_col: bool = True,
    ) -> None:
        """Initialize the DatasetSplitter class.
        Args:
            dataset: The dataset to split
            seed: The seed to use for the split

        Raises:
            Type: If the dataset is not a Dataset or a DatasetDict
            Exception: If the dataset is a DatasetDict and contains more than one column
        Note:
            The Dataset can be a DatasetDict or a Dataset.
            If it is a DatasetDict, it should contain only one column, and it will be converted as Dataset.
        """
        self._seed = seed
        self.append_split_col = append_split_col

    @property
    @abstractmethod
    def split_col_name(self) -> str:
        """Abstract method returning the name of the column containing the split name
        Returns:
            The name of the column containing the split name"""
        raise NotImplementedError
        return "awesome_split"

    @abstractmethod
    def _split(
        self,
    ) -> DatasetDict:
        """Implementing the dataset split method"""
        raise NotImplementedError

    def split(
        self, dataset: DatasetDict | Dataset, return_dataset_dict=True
    ) -> DatasetDict | Dataset:
        """Split the dataset and return it.
        Args:
            dataset: The dataset to split -
                if it is a DatasetDict, it should contain only one split
            return_dataset_dict: Whether to return a concatenated dataset or a DatasetDict
                If True -> return concatenated dataset -> Dataset
                If False, return a DatasetDict with the split name as key (Default)
        Return:
            The split dataset as a DatasetDict
        """

        self._return_dataset_dict = return_dataset_dict

        # Checking if the dataset is a Dataset or a DatasetDict of only one split
        self._dataset_check(dataset)

        # Makes sure that ._dataset is of type Dataset"""
        self._dataset_preprocessing(dataset)

        if self.split_col_name not in self._dataset.column_names:
            # if the split column is not in the dataset, create the split
            self._split_dataset = self._split()  # split the dataset
        else:
            self.append_split_col = False
            # if the split column is in the dataset, use it to split the dataset
            self._split_dataset = self.split_dataset_using_split_column(
                self._dataset, self.split_col_name
            )

        if self.append_split_col:
            # append the split column to the _split_dataset
            self._split_dataset = self._append_split_column(
                self._split_dataset, self.split_col_name
            )
            # update _dataset with the split column
            self._dataset = self.concatenated_dataset(self._split_dataset)

        return self.dataset

    @staticmethod
    def _dataset_check(dataset: DatasetDict | Dataset):
        """Checking if the dataset is a Dataset or a DatasetDict of only one split
        Args:
            dataset: The dataset to check
        Raises:
            TypeError: If the dataset is not a Dataset or a DatasetDict
            ValueError: If the dataset is a DatasetDict and contains more than one column
        """
        if not (isinstance(dataset, DatasetDict) or isinstance(dataset, Dataset)):
            raise TypeError(
                f"dataset must be a DatasetDict or a Dataset, got {type(dataset)} instead."
            )

        # if the dataset is a DatasetDict, convert it to a Dataset
        if isinstance(dataset, DatasetDict):
            # The input DatasetDict Should have only one split
            if len(dataset) > 1:
                raise ValueError(
                    f"A DatasetDict can be provided as an input, however it should contain only one split. "
                    f"Here the dataset contains {len(dataset)} splits: "
                    f"{[split for split in dataset]}"
                )

    def _dataset_preprocessing(self, dataset: Dataset | DatasetDict) -> None:
        """Makes sure that ._dataset is a Dataset"""
        if isinstance(dataset, DatasetDict):
            dataset = self.concatenated_dataset(dataset)

        self._dataset = dataset

    @property
    def dataset(self) -> Dataset | DatasetDict:
        """Return the dataset.
        Returns:
            The dataset as a DatasetDict or Dataset
        Note:
            If self.return_dataset_dict is True, return a DatasetDict
            If self.return_dataset_dict is False, return a Dataset
        """
        if self._return_dataset_dict:
            return self._split_dataset
        else:
            return self._dataset

    @staticmethod
    def split_dataset_using_split_column(
        dataset: Dataset, split_col_name: str
    ) -> DatasetDict:
        """Split the dataset using a column containing the split name.
        Args:
            dataset: The dataset to split
            split_col_name: The name of the column containing the split name
        Returns:
            DatasetDict containing the split dataset
        Raises:
            TypeError: If the dataset is not a Dataset
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be a Dataset, got {type(dataset)} instead.")

        split_dataset = DatasetDict()
        splits = sorted(list(set(dataset[split_col_name])))
        for split in splits:
            split_dataset[split] = dataset.filter(lambda x: x[split_col_name] == split)
        return split_dataset

    @staticmethod
    def concatenated_dataset(
        dataset: DatasetDict, split_list: list[str] = []
    ) -> Dataset:
        """Return the concatenated dataset.
        Args:
            dataset: The dataset to concatenate.
            split_list: The list of splits to concatenate
        Returns:
            The concatenated dataset
        """
        if not isinstance(dataset, DatasetDict):
            raise TypeError(
                f"dataset must be a DatasetDict, got {type(dataset)} instead."
            )

        if len(split_list) == 0:
            split_list = list(dataset.keys())

        return concatenate_datasets(
            [dataset[split] for split in dataset if split in split_list]
        )

    @staticmethod
    def _append_split_column(dataset: DatasetDict, split_col_name: str) -> DatasetDict:
        """Add a column to the dataset with the name of the split.
        Args:
            dataset: The dataset to modify
            split_col_name: The name of the column containing the split name
        Returns:
            The modified dataset
        """
        for split_name, dataset_split in dataset.items():
            # if the split column is already in the dataset, no need to add it again
            if split_col_name in dataset_split.column_names:
                break
            split_col = [split_name for _ in range(dataset_split.num_rows)]
            dataset[split_name] = dataset_split.add_column(split_col_name, split_col)

        return dataset

    @staticmethod
    def yield_all_train_test_sets(
        dataset: DatasetDict, combined_n: int = 1
    ) -> DatasetDict:
        """Yield all train test splits from a k-fold dataset
        Args:
            dataset: The dataset dictionnary to process
                It must contain items with keys such as "split_1", "split_2", ..., "split_k"
            combine_n: The number of splits to combine for the TEST set
        Yields:
            The train test splits
        Note:
            If the dataset is already a train test split, the train and test sets will simply be yielded

        Example:
            >>> import datasets
            >>> from pedata.preprocessing.split import DatasetSplitterRandomKFold
            >>> toy_dataset = datasets.Dataset.from_dict(
            ...    {
            ...         "aa_seq": ["MLGLYITR", "MAGLYITR", "MLYLYITR", "RAGLYITR", "MLRLYITR", "RLGLYITR"],
            ...         "target a": [1, 2, 3, 5, 4, 6],
            ...         "aa_mut": ["A2L", "wildtype", "A2L_G3Y", "M1R", "A2L_G3R", "A2L_M1R"],
            ...     }
            ... )
            >>> splitter = DatasetSplitterRandomKFold(toy_dataset, k=3)
            >>> dataset = splitter.split()
            >>> k = 0
            >>> for train_test_set in splitter.yield_all_train_test_sets(dataset, combined_n=1):
            ...     k += 1
            ...     print("________________")
            ...     print(f"- train_test_{k} -")
            ...     print(train_test_set)
            ________________
            - train_test_1 -
            DatasetDict({
                train: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 4
                })
                test: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 2
                })
            })
            ________________
            - train_test_2 -
            DatasetDict({
                train: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 4
                })
                test: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 2
                })
            })
            ________________
            - train_test_3 -
            DatasetDict({
                train: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 4
                })
                test: Dataset({
                    features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                    num_rows: 2
                })
            })
        """

        # Generate all combinations of combined_n elements from the list of splits
        dataset_splits = sorted(list(dataset.keys()))
        if dataset_splits != ["test", "train"]:
            combinations_list = list(combinations(dataset_splits, combined_n))
            # Yield all combinations as train test splits
            for combination in combinations_list:
                test = concatenate_datasets([dataset[j] for j in combination])
                train = concatenate_datasets(
                    [dataset[j] for j in dataset_splits if j not in combination]
                )
                yield DatasetDict({"train": train, "test": test})
        else:  # if the dataset is already a train test split, yield it
            yield dataset


class DatasetSplitterRandomTrainTest(DatasetSplitter):
    """Dataset Splitter for random Train Test Split."""

    def __init__(
        self,
        seed: int = 42,
    ) -> None:
        super().__init__(
            seed=seed,
        )

    @property
    def split_col_name(self) -> str:
        """Abstract method returning the name of the column containing the split name
        Retunrs:
            The name of the column containing the split name"""
        return "random_split_train_0_8_test_0_2"

    def _split(
        self,
    ) -> DatasetDict:
        """Split the dataset into train and test sets.
        Returns:
            The split dataset as a DatasetDict
        Example:
        >>> import datasets
        >>> toy_dataset = datasets.Dataset.from_dict(
        ...    {
        ...         "aa_seq": ["MLGLYITR", "MAGLYITR", "MLYLYITR", "RAGLYITR", "MLRLYITR"],
        ...         "target a": [1, 2, 3, 5, 4],
        ...         "aa_mut": ["A2L", "wildtype", "A2L_G3Y", "M1R", "A2L_G3R"],
        ...     }
        ... )
        >>> splitter = DatasetSplitterRandomTrainTest(toy_dataset)
        >>> dataset = splitter.split()
        >>> print(dataset)
        DatasetDict({
            train: Dataset({
                features: ['aa_seq', 'target a', 'aa_mut', 'random_split_train_0_8_test_0_2'],
                num_rows: 4
            })
            test: Dataset({
                features: ['aa_seq', 'target a', 'aa_mut', 'random_split_train_0_8_test_0_2'],
                num_rows: 1
            })
        })
        """
        return self._dataset.train_test_split(test_size=0.2, seed=self._seed)


class DatasetSplitterRandomKFold(DatasetSplitter):
    """Dataset Splitter for random Train Test Split."""

    def __init__(self, seed: int = 42, k: int = 10) -> None:
        super().__init__(
            seed=seed,
        )
        self.k = k

    @property
    def split_col_name(self) -> str:
        """Abstract method returning the name of the column containing the split name
        Retunrs:
            The name of the column containing the split name"""
        return f"random_split_{self.k}_fold"

    def _split(
        self,
    ) -> DatasetDict:
        """Split the dataset into K folds.
        Returns:
            The split dataset as a DatasetDict

        Raises:
            ValueError: If k is larger than the number of datapoints in the dataset

        Example:
        >>> import datasets
        >>> from pedata.preprocessing import DatasetSplitterRandomKFold
        >>> toy_dataset = datasets.Dataset.from_dict(
        ...    {
        ...         "aa_seq": ["MLGLYITR", "MAGLYITR", "MLYLYITR", "RAGLYITR", "MLRLYITR", "RLGLYITR"],
        ...         "target a": [1, 2, 3, 5, 4, 6],
        ...         "aa_mut": ["A2L", "wildtype", "A2L_G3Y", "M1R", "A2L_G3R", "A2L_M1R"],
        ...     }
        ... )
        >>> splitter = DatasetSplitterRandomKFold(toy_dataset, k=3)
        >>> dataset = splitter.split()
        >>> print(dataset)
        DatasetDict({
            split_0: Dataset({
                features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                num_rows: 2
            })
            split_1: Dataset({
                features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                num_rows: 2
            })
            split_2: Dataset({
                features: ['aa_seq', 'target a', 'aa_mut', 'random_split_3_fold'],
                num_rows: 2
            })
        })
        """
        if self.k > len(self._dataset):
            raise ValueError(
                f"k must be smaller than or equal to the number of datapoints in the dataset, "
                f"Here, we got k = {self.k} and {len(self._dataset)} datapoints respectively."
            )

        self._dataset = self._dataset.shuffle(seed=self._seed)  # shuffle the dataset
        fold_size = len(self._dataset) // self.k  # determine the fold size
        split_dataset = {}
        # Create k folds
        for i in range(self.k):
            # Determine the start and end index for the validation set
            start_index = i * fold_size
            end_index = (i + 1) * fold_size if i < self.k - 1 else len(self._dataset)
            split_dataset[f"split_{i}"] = Dataset.from_dict(
                self._dataset[start_index:end_index]
            )

        return DatasetDict(split_dataset)


def append_split_columns_to_dataset(dataset: Dataset) -> Dataset:
    """Append all split columns to a dataset
    Args:
        dataset: The dataset to modify
    Returns:
        The modified dataset
    """

    # Split the dataset into train and test sets
    dataset = DatasetSplitterRandomTrainTest().split(dataset, return_dataset_dict=False)

    # Split the dataset into k folds
    k = (
        10 if len(dataset) >= 10 else len(dataset)
    )  # hack for very small dataset (testing purposes)

    dataset = DatasetSplitterRandomKFold(k=k).split(dataset, return_dataset_dict=False)

    return dataset


if __name__ == "__main__":
    pass
