from pedata.preprocessing import append_index_column_to_dataset
from pedata.transform import transform_pipeline
import datasets
import pytest
from pytest import fixture


# FIXME use the conftest dataset_test once merged
@fixture(scope="module")
def dataset_as_Dataset() -> datasets.Dataset:
    """Return a toy dataset"""
    return datasets.Dataset.from_dict(
        {
            "aa_seq": [
                "MLGLYITR",
                "MAGLYITR",
                "MLYLYITR",
                "RAGLYITR",
                "MLRLYITR",
                "RLGLYITR",
            ],
            "target a": [1, 2, 3, 5, 4, 6],
            "aa_mut": [
                "A2L",
                "wildtype",
                "A2L_G3Y",
                "M1R",
                "A2L_G3R",
                "A2L_M1R",
            ],
        }
    )


@fixture(scope="module")
def dataset_as_DatasetDict(
    dataset_as_Dataset: datasets.Dataset,
) -> datasets.DatasetDict:
    """Return a toy dataset"""
    return datasets.DatasetDict({"whole_dataset": dataset_as_Dataset})


def test_append_index_column_to_dataset(dataset_as_Dataset):
    """Test that the append_index_column_to_dataset function works"""

    # ammends the dataset with an index column
    dataset_ammended = append_index_column_to_dataset(dataset_as_Dataset)
    assert "index" in dataset_ammended.column_names

    # does not ammend the dataset if it already has an index column"""
    dataset_ammended_2 = append_index_column_to_dataset(dataset_ammended)
    assert dataset_ammended == dataset_ammended_2


def test_append_index_column_to_dataset_with_incorrect_dataset(dataset_as_DatasetDict):
    """Test that the append_index_column_to_dataset returns TypeError if the input is incorrect"""
    with pytest.raises(TypeError):
        _ = append_index_column_to_dataset(dataset_as_DatasetDict)


def test_transform_pipeline(dataset_as_Dataset):
    """Test that the transform_pipeline function works"""
    # with defaults parameter
    dataset = transform_pipeline(dataset_as_Dataset)
    print(dataset)  # FIXME Write asserts and test with non defaults parameters


def test_transform_pipeline_with_incorrect_dataset(dataset_as_DatasetDict):
    """Test that the test_transform_pipeline returns TypeError if the input is incorrect"""
    with pytest.raises(TypeError):
        _ = transform_pipeline(dataset_as_DatasetDict)
