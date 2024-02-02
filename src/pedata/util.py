"""============ util.py ===========

The util module contains utility functions for HuggingFace datasets such as:
- data extraction
- normalization (z-score)
- optimization objective definition (min, max, fix)
- target adjustement for optimization objectives
- summary variable calculation
- metadata handling.

============ util.py ===========
"""
from typing import Callable, Sequence, Literal, Optional
from dataclasses import dataclass
import datasets as ds
from datasets import Dataset, load_dataset, concatenate_datasets
import numpy as np
import torch
from math import prod


__all__ = [
    "OptimizationObjective",
    "get_target_columns",
    "get_target",
    "adjust_all_targets_maximization",
    "zscore",
    "de_zscore_predictions",
    "get_summary_variable",
    "append_summary_variable",
    "DatasetHandler",
]


@dataclass
class OptimizationObjective:
    """Dataclass for optimization objectives.
    direction: Direction of the optimization objective.
        Valid values are 'min', 'max', and 'fix'. If 'fix', the value to aim for must be specified in `aim_for`.
    aim_for: Target value for the optimization objective. This is only used if `direction` is set to 'fix'.
    weight: Weight for the objective. This is used to adjust the relative importance of the objective compared to other objectives.

    Examples:
        >>> # For 'min' direction with default values for aim_for and weight
        >>> objective_min = OptimizationObjective(direction='min')
        >>> print(objective_min)
        OptimizationObjective(direction='min', aim_for=None, weight=1)

        >>> # For 'fix' direction with a specific aim_for value and default weight
        >>> objective_fix = OptimizationObjective(direction='fix', aim_for=42.0)
        >>> print(objective_fix)
        OptimizationObjective(direction='fix', aim_for=42.0, weight=1)

        >>> # For 'max' direction with custom values for aim_for and weight
        >>> objective_max = OptimizationObjective(direction='max', weight=2)
        >>> print(objective_max)
        OptimizationObjective(direction='max', aim_for=None, weight=2)
    """

    direction: Literal["max"] | Literal["min"] | Literal["fix"]
    aim_for: np.float16 | None = None  # target value for direction == 'fix'
    weight: np.uint8 = np.uint8(1)  # weight for the objective


def load_full_dataset(dataset_name: str) -> Dataset:
    """
    Load a full dataset rather than a specific split of the dataset
    Args:
        dataset_name: The name of the dataset
    Returns:
        full_dataset: The full dataset
    """
    dataset = load_dataset(dataset_name)
    full_dataset = concatenate_datasets([dataset[split] for split in dataset])
    return full_dataset


def get_target_columns(dataset: ds.Dataset) -> Sequence[str]:
    """
    Get target columns from dataset.

    Args:
        d (ds.Dataset): Dataset to extract target columns from

    Returns:
        Sequence[str]: Target columns

    Raises:
        TypeError: If the input is not a valid dataset object.

    Example:
    >>> import datasets as ds
    >>> dataset = ds.Dataset.from_dict({"aa_seq": ["MATCG", "KTGAC"], "target 1":[1, 2], "target 2": [3, 4]})
    >>> target_columns = get_target_columns(dataset)
    >>> print(target_columns)
    ['target 1', 'target 2']
    """

    if isinstance(dataset, ds.Dataset):
        # Extract target columns
        return [k for k in dataset.features if k.startswith("target")]

    else:
        raise TypeError(
            "Invalid input: try again with a valid dataset to extract target columns."
        )


def get_target(
    dataset: ds.Dataset,
    max_targets: int | None = None,
    normalization: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    as_dict: bool = False,
) -> dict[Sequence[str], np.ndarray] | tuple[Sequence[str], np.ndarray]:
    """Extract target variables from a Hugging Face dataset.

    This function retrieves the target variables from a Hugging Face dataset and provides options for normalization and returning the targets as a dictionary or tuple.

    Args:
        dataset: The dataset to extract target variables from.
        max_targets: Maximum number of targets to return. Defaults to None, which returns all targets.
        normalization: A function for normalizing the target values. Defaults to the identity function (no normalization).
        as_dict: Whether to return the targets as a dictionary of target names mapping to values. Defaults to False, in which case target names and values are returned as a tuple.

    Returns:
        If `as_dict` is False, returns a tuple containing the target names and the normalized target values.
        If `as_dict` is True, returns a dictionary where the keys are the target names and the values are the normalized target values.

    Example:
        >>> import datasets as ds
        >>> import numpy as onp
        >>> dataset = ds.Dataset.from_dict({"target1": onp.array([1, 2, 3]),"target2": onp.array([4, 5, 6]),"target3": onp.array([7, 8, 9])})
        >>> subset_targets_dict = get_target(dataset, max_targets=2, as_dict=True)

    """

    if not isinstance(dataset, ds.Dataset):
        raise TypeError("Invalid input: Please, input a valid HuggingFace Dataset")

    # Set the dataset format to pandas for easy column access
    dataset = dataset.with_format("pandas")

    targ_values = []  # Stores target values
    targ_keys = []  # Stores target keys

    # Iterate over target columns and extract values
    for targ in get_target_columns(dataset):
        targ_values.append(dataset[targ].values)
        targ_keys.append(targ)

        # Break the loop if the maximum number of targets is reached
        if max_targets is not None and len(targ_keys) >= max_targets:
            break

    # Convert target values to a stacked numpy array
    if len(targ_values) == 0:
        targ_values = np.array([])
    else:
        targ_values = np.vstack(targ_values).T

    # Add a new axis if the target values have only one dimension
    if len(targ_values.shape) == 1:
        targ_values = targ_values[:, np.newaxis]

    # Return targets as a tuple or dictionary, based on `as_dict` flag
    if not as_dict:
        return targ_keys, normalization(targ_values)
    else:
        return {k: normalization(targ_values.T[i]) for i, k in enumerate(targ_keys)}


def adjust_single_target_maximization(
    target_value: np.ndarray,
    target_opt_obj: OptimizationObjective,
) -> np.ndarray:
    """Adjust a single target variable to ensure that the optimization objective is always maximization.

    This function takes a single target variable `target_value` and an optimization objective `target_opt_obj`.
    It achieves this by performing either of the following steps for each target variable:
        1. Multiplying the target variable by -1 if the objective is minimization.
        2. Using the absolute difference between the target value and a fixed value to get close to the desired value if the objective is to fix the value.
        3. (default). Using the target variable as is if the objective is maximization.

    Args:
        target_value (np.ndarray): The target variable to adjust.
        target_opt_obj (OptimizationObjective): The optimization objective for the target variable.

    Returns:
        np.ndarray: The adjusted target variable.

    Raises:
        TypeError: If the input is not a valid OptimizationObjective object.
        ValueError: If the objective type is not one of the valid types.

    """
    if not isinstance(target_opt_obj, OptimizationObjective):
        raise TypeError(
            f"Invalid input: target_opt_obj must be of type OptimizationObjective. Received type {type(target_opt_obj)}."
        )

    if target_opt_obj.direction not in ("min", "max", "fix"):
        raise ValueError(
            f"Invalid objective type: {target_opt_obj.direction}. Expected value is 'min', 'max', or 'fix'."
        )

    if target_opt_obj.direction == "min":
        return (
            -target_value
        )  # Multiply the target variable by -1 if the objective is minimization

    elif target_opt_obj.direction == "max":
        return target_value  # Use the target variable as is if the objective is maximization

    else:
        return -np.abs(
            target_value - target_opt_obj.aim_for
        )  # Adjust the target variable to get close to a fixed value if the objective is fixed


def adjust_all_targets_maximization(
    dataset: ds.Dataset, objectives: dict[str, OptimizationObjective]
) -> ds.Dataset:
    """Adjust target variables to ensure that optimization objective is always maximization.

    Args:
        d: The dataset to adjust.
        objectives: Dictionary with targets names as keys and an OptimizationObjective Dataclass as values.
            see OptimizationObjective Dataclass for details.

    Returns:
        ds.Dataset: The adjusted dataset with modified target variables.

    Raises:
        Exception: If not all target columns have an objective specified.

    Notes:
        The objective type is specified as a dictionary with targets names as keys and an OptimizationObjective Dataclass as values.
        In this method, the only the objective direction is extracted from the OptimizationObjective Dataclass.

        Uses `adjust_single_target_maximization` nested method which takes a single target variable `target_value` and an optimization objective `target_opt_obj`.
        It adjusts the target by performing either of the following steps:
            1. Multiplying the target variable by -1 if the objective is minimization.
            2. Using the absolute difference between the target value and a fixed value to get close to the desired value if the objective is to fix the value.
            3. (default). Using the target variable as is if the objective is maximization.


    Example usage:
        >>> import datasets as ds
        >>> import pandas as pd
        >>> from pedata.util import adjust_all_targets_maximization, OptimizationObjective
        >>> dataset = ds.Dataset.from_pandas(pd.DataFrame({"target1": [11, 12, 13], "target2": [14, 15, 16], "target3": [7, 8, 9]}))
        >>> objectives = {"target1": OptimizationObjective(direction="min"), "target2": OptimizationObjective(direction="max"), "target3": OptimizationObjective(direction="fix", aim_for=10.0)}
        >>> adjusted_dataset = adjust_all_targets_maximization(dataset, objectives=objectives)
        >>> print(adjusted_dataset['target1'])
        [-11 -12 -13]
        >>> print(adjusted_dataset['target2'])
        [14 15 16]
        >>> print(adjusted_dataset['target3'])
        [-3 -2 -1]
    """
    # Get the list of target columns in the dataset
    target_cols = get_target_columns(dataset)

    # Check if objectives are specified for all target columns
    if set(target_cols) != set(objectives.keys()):
        raise Exception(
            f"All target columns must have an objective specified. Did not find objectives for {set(target_cols) - set(objectives.keys())}."
        )

    # Iterate and adjust target column values based on their objectives
    for target_property_name, target_opt_obj in objectives.items():
        # Adjust the target variables in the dataset based on the specified objectives
        dataset = (
            dataset.with_format()
            .map(
                lambda x: {
                    target_property_name: adjust_single_target_maximization(
                        np.array(x[target_property_name]), target_opt_obj
                    )
                },
                batched=True,
            )
            .with_format("numpy")  # Set the dataset format back to "numpy"
        )

    # Return the adjusted dataset
    return dataset


def zscore(
    array: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> np.ndarray:
    """Z-score is a statistical technique used to standardize an array
        This is by subtracting the mean and dividing by the standard deviation. This process transforms the
        values in the array to have a mean of 0 and a standard deviation of 1, providing a common scale for comparison.

    Args:
        array: The input array to be normalized.
        mean: Should be set as an input when the value is already known. Typically for transforming a test set/target
        std: Should be set as an input when the value is already known. Typically for transforming a test set/target

    Returns:
        np.ndarray: The normalized array.

    Examples:
        >>> import numpy as np
        >>> from pedata.util import zscore
        >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> zc_array = zscore(array)
        >>> print(zc_array) # doctest: +NORMALIZE_WHITESPACE
        [[-1.22474487 -1.22474487 -1.22474487]
        [ 0.          0.          0.        ]
        [ 1.22474487  1.22474487  1.22474487]]

    """

    if not isinstance(array, np.ndarray):
        raise TypeError("Invalid input: Input a valid numpy array")

    # Calculate the mean and standard deviation along axis 0
    if mean is None:
        mean = array.mean(0, keepdims=True)
    if std is None:
        std = array.std(0, keepdims=True)

    # Replace zero standard deviations with 1 to avoid division by zero
    std[std == 0] = 1

    # Normalize the array using the Z-score formula
    normalized_array = (array - mean) / std

    return normalized_array


def de_zscore_predictions(
    zs_pred_means: np.ndarray | None = None,
    zs_pred_stds: np.ndarray | None = None,
    mean: np.ndarray = np.array(0),
    std: np.ndarray = np.array(1),
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """de z-score a zcored array
    zs_pred_means can be either the mean or the std or both

    Args:
        zs_pred_means: The input mean array to be de_zscored.
        zs_pred_stds: The input std array to be de_zscored.
        mean: the mean from the original (train) array
        std: the std from the original (train) array

    Returns:
        np.ndarray: containing the de_zscored arrays:
            - the mean array if only mean is passed as an input
            - the mean and std arrays if both are passed as inputs
    """

    if not isinstance(zs_pred_means, np.ndarray):
        raise TypeError("Invalid input: zs_pred_means need to be a valid numpy array")

    de_zs_pred_means = zs_pred_means * std + mean

    if zs_pred_stds is not None:
        if not isinstance(zs_pred_stds, np.ndarray):
            raise TypeError(
                "Invalid input: zs_pred_stds need to be a valid numpy array"
            )
        de_zs_pred_stds = zs_pred_stds * std

        return de_zs_pred_means, de_zs_pred_stds

    else:
        return de_zs_pred_means


def get_summary_variable(
    dataset: ds.Dataset,
    normalization: Callable[[np.ndarray], np.ndarray] = zscore,
    objectives: dict[str, OptimizationObjective] | None = None,
) -> np.ndarray:
    """Extracts a summary variable from a dataset.
    Args:
        dataset: Dataset to extract summary variable from
        normalization: A normalization function for the summary variable. Defaults to zscore.
        objectives: The optimization objective for the target variable.

    Returns:
        np.ndarray: The summary variable.

    Raises:
        TypeError: If the input is not a valid dataset object.

    Notes:
        This function calculates a summary variable from a given dataset as follows:

        - If the objectives are specified, the summary variable is calculated as the weighted sum of the adjusted targets.
            - 0. Adjust the dataset targets to ensure that the optimization objective is always maximization.
            uses `adjust_all_targets_maximization` method
            - 1. Get the adjusted targets data from the dataset using the specified normalization function.
            - 2. Calculate the summary variable by computing the weigthed sum of the adjusted targets

        - If the objectives are not specified:
            - 0. Creates default objective (maximization, weight=1 for all targets) and adjust the dataset targets with it.
                Basically, this step does nothing to the target, but defines the objectives, which is a requirement for summing the targets, which requires the weights.
            - 1. Get the targets data from the dataset using the specified normalization function.
            - 2. Calculate the summary variable by computing the (weigthed) sum of the targets

    Example:
        >>> import numpy as np
        >>> import datasets as ds
        >>> from pedata.util import OptimizationObjective, get_summary_variable
        >>> dataset = ds.Dataset.from_dict({"target1": np.arange(10) + 1, "target2": np.arange(10),"target3": np.arange(10)})
        >>> normalised_summary = get_summary_variable(dataset)
        >>> print(normalised_summary) # doctest: +NORMALIZE_WHITESPACE
        [-4.70009671 -3.65563078 -2.61116484 -1.5666989  -0.52223297  0.52223297
        1.5666989   2.61116484  3.65563078  4.70009671]
    """

    # Validate input
    if not isinstance(dataset, ds.Dataset):
        raise TypeError("Invalid input: Please, input a valid HuggingFace Dataset")

    # 0 - Set default objectives if not specified
    if objectives is None:
        objectives = {}
        target_names = get_target_columns(dataset)
        for target_name in target_names:
            objectives[target_name] = OptimizationObjective(direction="max")

    # 0 - Adjust the dataset according to the objectives
    dataset = adjust_all_targets_maximization(dataset, objectives=objectives)

    # 1 - Get the target data from the dataset using the specified normalization function
    target_data = get_target(dataset, normalization=normalization, as_dict=True)

    # 2 - Calculate the summary variable by summing the (adjusted) targets
    summary = np.zeros(dataset.num_rows, dtype=np.float64)

    for target_name, target_value in target_data.items():
        summary += target_value * objectives[target_name].weight

    # Return the summary variable
    return summary


def append_summary_variable(
    dataset: ds.Dataset,
    normalization: Callable[[np.ndarray], np.ndarray] = zscore,
    objectives: dict[str, OptimizationObjective] | None = None,
    summary_variable_name: str = "target summary variable",
) -> ds.Dataset:
    """Append summary variable to a dataset.
    Args:
        dataset: Dataset to append summary variable to
        normalization: A normalization function for the summary variable. Defaults to zscore.
        objectives: The optimization objective for the target variable.

    Returns:
        Dataset with summary variable appended as a new column.

    Raises:
        ValueError: If the dataset contains no target columns.

    Example:
        >>> import numpy as np
        >>> import datasets as ds
        >>> from pedata.util import OptimizationObjective, append_summary_variable, zscore
        >>> dataset = ds.Dataset.from_dict({
        ...     "target1": np.arange(10, dtype=np.float64),
        ...     "target2": np.random.random(10),
        ...     "target3": np.arange(10, dtype=np.float64)[::-1],
        ... })
        >>> objectives = {
        ...     "target1": OptimizationObjective(direction="min", weight=2),
        ...     "target2": OptimizationObjective(direction="max", weight=1),
        ...     "target3": OptimizationObjective(direction="fix", aim_for=0.5, weight=10),
        ... }
        >>> dataset = append_summary_variable(
        ...     dataset,
        ...     normalization=zscore,
        ...     objectives=objectives,
        ...     summary_variable_name="target summary variable",
        ... )
        >>> print(dataset) # doctest: +NORMALIZE_WHITESPACE
        Dataset({
            features: ['target1', 'target2', 'target3', 'target summary variable'],
            num_rows: 10
        })
    """
    num_targets = len(list(get_target_columns(dataset)))
    if num_targets < 1:
        raise ValueError("Dataset contains no target columns.")
    else:
        return dataset.add_column(
            summary_variable_name,
            get_summary_variable(
                dataset, normalization=normalization, objectives=objectives
            ),
        )


class DatasetHandler(object):
    """Class for handling datasets in HuggingFace format.
    It stores the feature name, type, and dimension information for each feature as metadata.
    it allows to extract features from the dataset and concatenate them into a single tensor
    """

    def __init__(
        self,
        hf_ds: ds.Dataset | dict,
        features: Optional[list[str]],
        **kwargs,
    ):
        """
        Generates the metadata for the dataset and stores it in the object.
        Args:
            hf_ds : Dataset to handle. If a dictionary is passed, it is converted to a Dataset object.
            features : List of feature names to handle

        """
        if not isinstance(hf_ds, Dataset):
            # convert to dataset if not already
            # (before converting to dict, necessary when using dataset.map())
            hf_ds = Dataset.from_dict(dict(hf_ds))
        hf_ds = hf_ds.with_format("torch")
        if features is None:
            print("No features specified, using all features in dataset")
            features = list(hf_ds.features)
        self.features = features
        self.metadata = {
            k: {"type": hf_ds[k].dtype, "shape": hf_ds[k].shape} for k in features
        }
        # compute the start and stop indices for each feature in the concatenated tensor
        self.idx = {}

        for feat in self.features:
            # compute start index
            start_idx = 0
            for k in self.features:
                if k != feat:
                    if len(self.metadata[k]["shape"]) > 1:
                        start_idx += prod(self.metadata[k]["shape"][1:])
                    else:
                        start_idx += 1
                else:
                    break
            if len(self.metadata[feat]["shape"]) > 1:
                end_idx = start_idx + prod(self.metadata[feat]["shape"][1:])
            else:
                end_idx = start_idx + 1
            self.idx[feat] = slice(start_idx, end_idx)

    def cat(self, hf_ds: Dataset) -> torch.Tensor:
        """Return concatenated tensor with all features and most general type.
        Args:
            hf_ds: Dataset to concatenate

        Returns:
            Concatenated tensor

        Example:
            >>> import datasets as ds
            >>> dataset = ds.Dataset.from_dict({"aa_seq": ["MATCG", "KTGAC"], "target_1":[1, 2], "target_2": [3, 4]})
            >>> hfds_metadata = DatasetHandler(dataset, ["target_1", "target_2"])
            >>> print(hfds_metadata.cat(dataset))
            tensor([[1., 3.],
                    [2., 4.]], dtype=torch.float64)

        """
        if not isinstance(hf_ds, Dataset):
            # convert to dataset if not already
            # (before converting to dict, necessary when using dataset.map())
            hf_ds = Dataset.from_dict(dict(hf_ds))

        hf_ds = hf_ds.with_format("torch")
        conc_tensor = torch.cat(
            [
                hf_ds[k].reshape(len(hf_ds[k]), -1)
                if len(self.metadata[k]["shape"]) > 1
                else hf_ds[k][:, None]
                for k in self.features
            ],
            dim=-1,
        ).to(torch.float64)
        return conc_tensor

    def get(self, t: torch.Tensor, *feat: list[str]) -> torch.Tensor:
        """Return the feature(s) in the concatenated tensor in concatenation axis (i.e. the last axis).
        Args:
            t: Concatenated tensor
            feat: List of feature names to extract

        Returns:
            Tensor containing the extracted feature(s)

        Example:
            >>> import datasets as ds
            >>> dataset = ds.Dataset.from_dict({"aa_seq": ["MATCG", "KTGAC"], "target 1":[1, 2], "target 2": [3, 4]})
            >>> hfds_metadata = DatasetHandler(dataset, ["target 1", "target 2"])
            >>> print(hfds_metadata.get(hfds_metadata.cat(dataset), ["target 1"]))
            tensor([[1.],
                    [2.]], dtype=torch.float64)

        """
        return t[..., self.dims(feat)]

    def dims(self, feat: list[str]) -> tuple[int]:
        """Return the dimensions of the feature(s) in the concatenated tensor in concatenation axis (i.e. the last axis).

        Args:
            feat: List of feature names.

        Returns:
            Tuple containing the dimensions of the feature(s) in the concatenated tensor.

        Example:
            >>> import datasets as ds
            >>> dataset = ds.Dataset.from_dict({"aa_seq": ["MATCG", "KTGAC"], "target 1":[1, 2], "target 2": [[3, 4], [5, 6]]})
            >>> hfds_metadata = DatasetHandler(dataset, ["target 1", "target 2"])
            >>> print(hfds_metadata.dims(["target 2", "target 1"]))
            (1, 2, 0)
        """
        rval = []
        for f in feat:
            rval.extend(list(range(self.idx[f].start, self.idx[f].stop)))
        return tuple(rval)


if __name__ == "__main__":
    pass
