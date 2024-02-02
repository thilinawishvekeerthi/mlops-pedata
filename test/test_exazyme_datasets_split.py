import datasets
import pytest
from pytest import fixture
from pedata.preprocessing import (
    DatasetSplitterRandomTrainTest,
    DatasetSplitterRandomKFold,
)


# toy dataset
dataset = datasets.Dataset.from_dict(
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


# FIXME use the conftest dataset_test once merged add (scope="session") to the fixture - so the dataset does not get remade for each test
@fixture(scope="module")
def toy_dataset() -> datasets.Dataset:
    """Return a toy dataset"""
    return dataset


@fixture
def toy_dataset_as_DatasetDict() -> datasets.DatasetDict:
    """Return a toy dataset as a DatasetDict"""
    return datasets.DatasetDict({"all": dataset})


def test_dataset_splitter_takes_Dataset(toy_dataset):
    """Test that the DatasetSplitterRandomTrainTest class can take a Dataset as input"""
    _ = DatasetSplitterRandomTrainTest().split(toy_dataset, return_dataset_dict=True)


def test_dataset_splitter_takes_DatasetDict(toy_dataset_as_DatasetDict):
    """Test that the DatasetSplitterRandomTrainTest class can take a DatasetDict as input"""
    _ = DatasetSplitterRandomTrainTest().split(
        toy_dataset_as_DatasetDict, return_dataset_dict=True
    )


def test_dataset_splitter_returns_Dataset(toy_dataset):
    """Test that the DatasetSplitterRandomTrainTest class returns a Dataset when return_dataset_dict=False"""
    train_test_ds = DatasetSplitterRandomTrainTest().split(
        toy_dataset, return_dataset_dict=False
    )

    assert isinstance(train_test_ds, datasets.Dataset)


def test_dataset_splitter_returns_DatasetDict(toy_dataset):
    """Test that the DatasetSplitterRandomTrainTest class returns a DatasetDict when return_dataset_dict=True"""
    train_test_ds = DatasetSplitterRandomTrainTest().split(toy_dataset)

    assert isinstance(train_test_ds, datasets.DatasetDict)


def test_dataset_splitter_random_train_tes_split_set_seed(toy_dataset):
    """Test that the DatasetSplitterRandomTrainTest class returns the same DatasetDict when the seed is set
    Test that setting the seed with a given value always returns the same DatasetDict as output
    """
    train_test_ds = DatasetSplitterRandomTrainTest(
        seed=93,
    ).split(toy_dataset, return_dataset_dict=True)

    assert isinstance(train_test_ds, datasets.DatasetDict)

    train_test_ds["test"]["aa_seq"]
    assert train_test_ds["test"]["aa_seq"] == [
        "RAGLYITR",
        "MAGLYITR",
    ], f"seed 93; train_test_ds['test']['aa_seq'] = {train_test_ds['test']['aa_seq']} but should be ['RAGLYITR', 'MAGLYITR']"


def test_dataset_splitter_concatenating_only_one_split(toy_dataset):
    """Test that the DatasetSplitterRandomTrainTest class returns the same DatasetDict when the seed is set
    Test that setting the seed with a given value always returns the same DatasetDict as output
    """
    splitter = DatasetSplitterRandomTrainTest(
        seed=93,
    )
    dataset = splitter.split(toy_dataset, return_dataset_dict=True)
    assert isinstance(dataset, datasets.DatasetDict)
    dataset_cat = splitter.concatenated_dataset(dataset, split_list=["train"])
    assert dataset_cat["aa_seq"] == ["MLYLYITR", "RLGLYITR", "MLRLYITR", "MLGLYITR"]


def test_input_improper_datasetdict(toy_dataset):
    """Test that the DatasetSplitterRandomTrainTest class returns the same DatasetDict when the seed is set
    Test that setting the seed with a given value always returns the same DatasetDict as output
    """
    splitter = DatasetSplitterRandomTrainTest()
    dataset_dict = splitter.split(toy_dataset, return_dataset_dict=True)
    with pytest.raises(ValueError):
        splitter = DatasetSplitterRandomTrainTest().split(dataset_dict)


def test_dataset_splitter_k_fold(toy_dataset):
    """test that the DatasetSplitterRandomKFold class returns the correct number of splits"""
    splitter = DatasetSplitterRandomKFold(k=3)
    dataset = splitter.split(toy_dataset, return_dataset_dict=True)

    assert len(dataset) == 3
    assert dataset["split_0"]["aa_seq"] == ["RAGLYITR", "MLYLYITR"]


def test_dataset_splitter_k_fold_splits_are_already_there(toy_dataset):
    """test that the DatasetSplitterRandomKFold class returns the correct number of splits"""
    splitter = DatasetSplitterRandomKFold(k=3)
    dataset = splitter.split(toy_dataset, return_dataset_dict=False)

    splitter2 = DatasetSplitterRandomKFold(k=3)
    dataset2 = splitter2.split(toy_dataset, return_dataset_dict=False)
    assert (
        dataset2["target a"] == dataset["target a"]
        and dataset2["aa_seq"] == dataset["aa_seq"]
    )


def test_dataset_splitter_k_fold_yielf_train_tes_sets(toy_dataset):
    """Test that the DatasetSplitterRandomKFold class yields the correct train test sets"""
    splitter = DatasetSplitterRandomKFold(k=3)
    dataset = splitter.split(toy_dataset)
    k = 0
    for train_test_set in splitter.yield_all_train_test_sets(dataset):
        if k == 0:
            assert train_test_set["train"]["aa_seq"] == [
                "RLGLYITR",
                "MLRLYITR",
                "MAGLYITR",
                "MLGLYITR",
            ]
        elif k == 1:
            assert train_test_set["train"]["aa_seq"] == [
                "RAGLYITR",
                "MLYLYITR",
                "MAGLYITR",
                "MLGLYITR",
            ]

        k += 1


def test_dataset_splitter_k_fold_yielf_train_tes_sets_2(toy_dataset):
    """Test that the DatasetSplitterRandomKFold class yields the correct train test sets"""
    splitter = DatasetSplitterRandomTrainTest()
    dataset = splitter.split(toy_dataset)
    k = 0
    for train_test_set in splitter.yield_all_train_test_sets(dataset):
        assert train_test_set["train"]["aa_seq"] == [
            "RLGLYITR",
            "MLRLYITR",
            "MAGLYITR",
            "MLGLYITR",
        ]
        k += 1

    assert k == 1


def test_dataset_splitter_k_fold_yielf_train_tes_sets_combined_n(toy_dataset):
    """Test that the DatasetSplitterRandomKFold class yields the correct train test sets when combined_n=2"""
    splitter = DatasetSplitterRandomKFold(k=6)
    dataset = splitter.split(toy_dataset)

    for split_n, split in enumerate(
        splitter.yield_all_train_test_sets(dataset, combined_n=2)
    ):
        if split_n == 0:
            assert split["test"]["aa_seq"][:2] == [
                "RAGLYITR",
                "MLYLYITR",
            ]
        elif split_n == 1:
            assert split["train"]["aa_seq"][:2] == ["MLYLYITR", "MLRLYITR"]
