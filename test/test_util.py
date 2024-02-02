import numpy as np
import pandas as pd
import torch
import datasets as ds

from pedata.util import (
    OptimizationObjective,
    DatasetHandler,
    adjust_all_targets_maximization,
    get_summary_variable,
    append_summary_variable,
    get_target_columns,
    get_target,
    zscore,
)
from pedata.config import add_encodings

import pytest
from pytest import fixture

from pedata import (
    aa_example_0_no_missing_val,
)


def get_small_large_df() -> tuple[ds.Dataset, ds.Dataset, str]:
    base_seq = "MAGLYITR"
    large_df = ds.Dataset.from_dict(
        {
            "aa_seq": [
                "MLGLYITR",
                base_seq,
                "MLYLYITR",
            ],
            "target a": [1, 2, 3],
            "aa_mut": ["A2L", "wildtype", "A2L_G3Y"],
        }
    )

    small_df = ds.Dataset.from_dict(
        {
            "aa_seq": [
                "MLGLYITR",
                base_seq,
            ],
            "target a": [
                1,
                2,
            ],
            "aa_mut": ["A2L", "wildtype"],
        }
    )

    large_df = add_encodings(large_df)
    small_df = add_encodings(small_df)
    for df in [large_df, small_df]:
        df = df.with_format(columns=["aa_seq", "aa_mut"])
    return small_df, large_df, base_seq


def test_get_target_columns():
    # Test case 1: Non-dataset Object
    invalid_input = {"aa_seq": ["MAPEKT"], "target foo": {1}}
    with pytest.raises(TypeError):
        get_target_columns(invalid_input)

    # Test case 2: Empty Dataset
    dataset = ds.Dataset.from_dict({"aa_seq": [], "target 1": [], "target 2": []})
    assert get_target_columns(dataset) == ["target 1", "target 2"]

    # Test case 3: Dataset with no target column
    dataset = ds.Dataset.from_dict(
        {"aa_mut": ["wt", "A2G"], "aa_seq": ["MAPEKT", None]}
    )
    assert get_target_columns(dataset) == []

    # Test case 4: Dataset with a single target column
    dataset = ds.Dataset.from_dict(aa_example_0_no_missing_val)
    assert get_target_columns(dataset) == ["target foo"]

    # Test case 5: Dataset with multiple target columns
    dataset = ds.Dataset.from_dict(
        {
            "dna_seq": ["GCTATC", "AATCCG"],
            "target 1": [1, 2],
            "target 2": [3, 4],
            "target 3": [5, 6],
        }
    )
    assert get_target_columns(dataset) == ["target 1", "target 2", "target 3"]


def test_get_target():
    # Test case 1: Invalid input
    invalid_input = {"target": [1, 2]}
    with pytest.raises(TypeError):
        get_target(invalid_input)

    # Test case 2: Return all targets as a tuple
    dataset = ds.Dataset.from_dict(
        {
            "target1": np.array([1, 2, 3]),
            "target2": np.array([4, 5, 6]),
            "target3": np.array([7, 8, 9]),
        }
    )
    targets_tuple = get_target(dataset)
    assert isinstance(targets_tuple, tuple) and targets_tuple[0] == [
        "target1",
        "target2",
        "target3",
    ]

    # Test case 3: Return all targets as a dictionary
    targets_dict = get_target(dataset, as_dict=True)
    expected_targets_dict = {
        "target1": np.array([1, 2, 3]),
        "target2": np.array([4, 5, 6]),
        "target3": np.array([7, 8, 9]),
    }
    for key in targets_dict:
        assert np.array_equal(targets_dict[key], expected_targets_dict[key])

    # Test case 4: Return a limited number of targets
    subset_targets_tuple = get_target(dataset, max_targets=2)
    assert isinstance(subset_targets_tuple, tuple) and subset_targets_tuple[0] == [
        "target1",
        "target2",
    ]

    # Test case 5: Return a subset of targets as a dictionary
    subset_targets_dict = get_target(dataset, max_targets=2, as_dict=True)
    expected_targets_dict = {
        "target1": np.array([1, 2, 3]),
        "target2": np.array([4, 5, 6]),
    }
    for key in expected_targets_dict:
        assert np.array_equal(subset_targets_dict[key], expected_targets_dict[key])

    # Test case 6: Apply normalization zscore function
    target_dict = get_target(dataset, normalization=zscore, as_dict=True)
    expected_targets_dict = {
        "target1": np.array([-1.224744871391589, 0.0, 1.224744871391589]),
        "target2": np.array([-1.224744871391589, 0.0, 1.224744871391589]),
        "target3": np.array([-1.224744871391589, 0.0, 1.224744871391589]),
    }
    for key in target_dict:
        assert np.array_equal(target_dict[key], expected_targets_dict[key])


def test_adjust_all_targets_maximization():
    # Test Case 1: Minimization objective
    dataset = ds.Dataset.from_pandas(pd.DataFrame({"target1": [1, 2, 3]}))
    objectives = {"target1": OptimizationObjective(direction="min")}
    adjusted_dataset = adjust_all_targets_maximization(dataset, objectives)
    assert np.all(adjusted_dataset["target1"] == [-1, -2, -3])

    # Test Case 2: Maximization objective
    dataset = ds.Dataset.from_pandas(pd.DataFrame({"target2": [4, 5, 6]}))
    objectives = {"target2": OptimizationObjective(direction="max")}
    adjusted_dataset = adjust_all_targets_maximization(dataset, objectives)
    assert np.all(adjusted_dataset["target2"] == [4, 5, 6])

    # Test Case 3: Fixed value objective
    dataset = ds.Dataset.from_pandas(pd.DataFrame({"target3": [7, 8, 9]}))
    objectives = {"target3": OptimizationObjective(direction="fix", aim_for=10)}
    adjusted_dataset = adjust_all_targets_maximization(dataset, objectives)
    assert np.all(adjusted_dataset["target3"] == [-3, -2, -1])

    # Test Case 4: Multiple target columns with different objectives
    dataset = ds.Dataset.from_pandas(
        pd.DataFrame({"target4": [11, 12, 13], "target5": [14, 15, 16]})
    )
    objectives = {
        "target4": OptimizationObjective(direction="min"),
        "target5": OptimizationObjective(direction="max"),
    }

    adjusted_dataset = adjust_all_targets_maximization(dataset, objectives)
    assert np.all(adjusted_dataset["target4"] == [-11, -12, -13]) and np.all(
        adjusted_dataset["target5"] == [14, 15, 16]
    )

    # Test Case 5: Missing objective for a target column
    dataset = ds.Dataset.from_pandas(pd.DataFrame({"target6": [17, 18, 19]}))
    objectives = {
        "target7": OptimizationObjective(direction="min")
    }  # Missing objective for "target6"
    with pytest.raises(Exception):
        adjusted_dataset = adjust_all_targets_maximization(dataset, objectives)

    # Test Case 6: Invalid objective type
    dataset = ds.Dataset.from_pandas(pd.DataFrame({"target8": [20, 21, 22]}))
    objectives = {"target8": OptimizationObjective(direction="invalid")}
    with pytest.raises(ValueError):
        adjusted_dataset = adjust_all_targets_maximization(dataset, objectives)

    # Test Case 7: Apply Maximization, Minimization, and Fixed Value Objectives
    dataset = ds.Dataset.from_dict(aa_example_0_no_missing_val)
    columns = get_target_columns(dataset)
    dataset = dataset.with_format("numpy")

    # Add additional columns to the dataset and define objectives for target columns
    dataset = dataset.add_column("target minimize", np.arange(len(dataset))).add_column(
        "target fix", np.arange(len(dataset))
    )
    objectives = {
        "target minimize": OptimizationObjective(direction="min"),
        "target fix": OptimizationObjective(direction="fix", aim_for=len(dataset)),
    }

    # Set maximization objective for all other target columns
    for col in columns:
        objectives[col] = OptimizationObjective(direction="max")

    # Apply target variable adjustment and set adjusted dataset to "numpy" for assertions
    adjusted_dataset = adjust_all_targets_maximization(dataset, objectives)
    adjusted_dataset = adjusted_dataset.with_format("numpy")

    # Assert that maximization columns are unaltered
    for c in columns:
        assert np.all(
            adjusted_dataset[c] == dataset[c]
        ), "Maximization columns should be unaltered."

    # Assert that minimization columns are negated
    assert np.all(
        -adjusted_dataset["target minimize"] == np.arange(len(dataset))
    ), "Minimization columns should be negated."

    # Assert that fixed target values are non-negative
    assert np.all(
        adjusted_dataset["target fix"] <= 0
    ), "Fixed target values should be non-negative."

    # Assert that fixed target values are correct
    assert np.all(
        adjusted_dataset["target fix"]
        == -np.abs(np.arange(len(dataset)) - len(dataset))
    ), "Fixed target values are off."


def test_zscore():
    # Test case 1: 1D array of consecutive numbers from 1 to 9
    input_array = np.arange(10)
    expected_output = zscore(input_array)
    assert np.allclose(expected_output.mean(), np.zeros_like(input_array), atol=1e-5)
    assert np.allclose(expected_output.std(), np.ones_like(input_array), atol=1e-5)

    # Test case 2: Basic example with positive values
    input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_output = np.array(
        [
            [-1.22474487, -1.22474487, -1.22474487],
            [0.0, 0.0, 0.0],
            [1.22474487, 1.22474487, 1.22474487],
        ]
    )
    assert np.allclose(zscore(input_array), expected_output)

    # Test case 3: Array with negative values
    input_array = np.array([[-2, -4, -6], [0, 2, 4], [6, 8, 10]])
    expected_output = np.array(
        [
            [-0.98058068, -1.22474487, -1.31319831],
            [-0.39223227, 0.0, 0.20203051],
            [1.37281295, 1.22474487, 1.1111678],
        ]
    )
    assert np.allclose(zscore(input_array), expected_output)

    # Test case 4: Array with all zeros
    input_array = np.zeros((3, 3))
    expected_output = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(zscore(input_array), expected_output)

    # Test case 5: Array with only one element
    input_array = np.array([[5]])
    expected_output = np.array([[0.0]])
    assert np.allclose(zscore(input_array), expected_output)

    # Test case 6: Array with all identical values
    input_array = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
    expected_output = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(zscore(input_array), expected_output)

    # Test case 7: Array with single row
    input_array = np.array([[1, 2, 3]])
    expected_output = np.array([[0.0, 0.0, 0.0]])
    assert np.allclose(zscore(input_array), expected_output)

    # Test case 8: Array with single column
    input_array = np.array([[1], [2], [3]])
    expected_output = np.array([[-1.22474487], [0.0], [1.22474487]])
    assert np.allclose(zscore(input_array), expected_output)

    # Test case 9: Invalid Input
    invalid_input = [1, 2, 4]
    with pytest.raises(TypeError):
        zscore(invalid_input)


def test_get_summary_variable():
    # Test case 1: Unnormalized summary variable
    fix = np.abs(np.arange(10) - 4.5)
    dataset = ds.Dataset.from_dict(
        {
            "target max": np.arange(10) + 1,
            "target min": np.arange(10),
            "target fix": np.arange(10),
        }
    )
    objectives = {
        "target max": OptimizationObjective(direction="max", weight=1),
        "target min": OptimizationObjective(direction="min", weight=1),
        "target fix": OptimizationObjective(direction="fix", aim_for=4.5, weight=1),
    }
    # Since target max - target min = 1, the unnormalized summary variable should be fix + 1
    unnorm_summary = get_summary_variable(
        dataset, normalization=lambda x: x, objectives=objectives
    )
    assert np.all(unnorm_summary == 1 - fix)

    # Test case 2: Normalized summary variable
    norm_summary = get_summary_variable(
        dataset, normalization=zscore, objectives=objectives
    )
    assert np.allclose(
        norm_summary,
        get_summary_variable(
            adjust_all_targets_maximization(dataset, objectives=objectives), zscore
        ),
        atol=1e-5,
    )

    # Test case 3: With no objectives
    normalised_summary = get_summary_variable(dataset)
    expected_summary = np.array(
        [
            -4.70009671,
            -3.65563078,
            -2.61116484,
            -1.5666989,
            -0.52223297,
            0.52223297,
            1.5666989,
            2.61116484,
            3.65563078,
            4.70009671,
        ]
    )
    assert np.allclose(normalised_summary, expected_summary)

    # Test case 4: Invalid input
    invalid_input = {"target1": np.arange(10), "target2": np.arange(10) + 1}
    with pytest.raises(TypeError):
        get_summary_variable(invalid_input)


def test_append_summary_variable():
    dataset = ds.Dataset.from_dict(
        {
            "target1": np.arange(10, dtype=np.float64),
            "target2": np.random.random(10),
            "target3": np.arange(10, dtype=np.float64)[::-1],
        }
    )
    objectives = {
        "target1": OptimizationObjective(direction="min", weight=2),
        "target2": OptimizationObjective(direction="max", weight=1),
        "target3": OptimizationObjective(direction="fix", aim_for=0.5, weight=10),
    }
    dataset = append_summary_variable(
        dataset,
        normalization=zscore,
        objectives=objectives,
        summary_variable_name="target summary variable",
    )


def test_regression_adjust_target_maximization():
    objectives = {"target a": OptimizationObjective(direction="max")}
    small_df, _, _ = get_small_large_df()
    adjust_all_targets_maximization(small_df, objectives=objectives)


def test_OptimizationObjective():
    objective_min = OptimizationObjective(direction="min")

    # Check that the min value is set correctly
    assert (
        objective_min.direction == "min"
        and objective_min.aim_for is None
        and objective_min.weight == 1
    )

    # Check that the fix value is set correctly
    objective_fix = OptimizationObjective(direction="fix", aim_for=42.0)
    assert (
        objective_fix.direction == "fix"
        and objective_fix.aim_for == 42.0
        and objective_fix.weight == 1
    )

    # Check that the max value is set correctly
    objective_max = OptimizationObjective(direction="max", weight=2)
    assert (
        objective_max.direction == "max"
        and objective_min.aim_for is None
        and objective_max.weight == 2
    )

    # Check that the direction value is set
    with pytest.raises(TypeError):
        _ = OptimizationObjective()


# ======= DATASET HANDLER ========
@fixture
def single_feature_dataset():
    return ds.Dataset.from_dict({"target 1": [1, 2]})


@fixture
def single_expected_output():
    return torch.tensor([[1.0], [2.0]], dtype=torch.float64)


@fixture
def hfds_metadata_single(single_feature_dataset):
    return DatasetHandler(single_feature_dataset, ["target 1"])


def test_DatasetHandler_single_feature(
    single_feature_dataset, single_expected_output, hfds_metadata_single
):
    # Test case 1: Test cat() method with a dataset containing a single feature
    assert torch.allclose(
        hfds_metadata_single.cat(single_feature_dataset), single_expected_output
    )

    # Test case 2: Test get() method with a single feature
    conc_tensor = hfds_metadata_single.cat(single_feature_dataset)
    assert torch.allclose(
        hfds_metadata_single.get(conc_tensor, "target 1"), single_expected_output
    )

    # Test case 3: Test dims() method with a single feature
    assert hfds_metadata_single.dims(["target 1"]) == (0,)


@fixture
def multi_feature_dataset():
    return ds.Dataset.from_dict(
        {"aa_seq": ["MATCG", "KTGAC"], "target 1": [1, 2], "target 2": [3, 4]}
    )


@fixture
def multi_expected_output():
    return torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float64)


@fixture
def hfds_metadata_multi(multi_feature_dataset):
    return DatasetHandler(multi_feature_dataset, ["target 1", "target 2"])


def test_DatasetHandler_multi_feature(
    multi_feature_dataset, multi_expected_output, hfds_metadata_multi
):
    # Test case 1: Test cat() method with a dataset containing multiple features
    assert torch.allclose(
        hfds_metadata_multi.cat(multi_feature_dataset), multi_expected_output
    )

    # Test case 2: Test get() method with multiple features
    conc_tensor = hfds_metadata_multi.cat(multi_feature_dataset)
    assert torch.allclose(
        hfds_metadata_multi.get(conc_tensor, "target 1", "target 2"),
        multi_expected_output,
    )

    # Test case 3: Test dims() method with multiple features

    assert hfds_metadata_multi.dims(["target 1", "target 2"]) == (0, 1)
