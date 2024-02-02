import pandas as pd
from datasets import Dataset
import os
from pathlib import Path
import pytest
from pedata.disk_cache import (
    preprocess_data,
    load_similarity,
    get_missing_values,
    fill_missing_sequences,
    hfds_from_pydict,
    save_dataset_as_csv,
    read_dataset_from_file,
    _read_csv_ignore_case,
)

from pedata import (
    dataset_dict_regression,
    dna_example_1_missing_val,
    dna_example_1_no_missing_val,
    aa_example_0_missing_val,
    aa_example_0_no_missing_val,
    aa_example_0_invalid_alphabet,
)
import fsspec
import shutil
from pytest import fixture


# ========= Helper functions ======
def clean_up() -> None:
    """Delete created temporarly files and folders"""
    file_list = [
        "tempfile.csv",
        "tEsT_DATA.CsV",
        "tempfile.xls",
        "tempfile.xlsx",
        "processed_data",
        "tempfile.html",
    ]
    for file in file_list:
        if file == "processed_data":
            shutil.rmtree(file, ignore_errors=True)
        if os.path.exists(file):
            os.remove(file)


# ======== Fixtures ======
@fixture(scope="module")
def aa_expected_new_features():
    return [
        "aa_unirep_1900",
        "aa_unirep_final",
        "aa_len",
        "aa_1gram",
        "aa_1hot",
    ]


@fixture(scope="module")
def dna_expected_new_features():
    return ["dna_len", "dna_1hot"]


@fixture(scope="module")
def alphabet_type_dna():
    return "dna"


@fixture(scope="module")
def alphabet_type_aa():
    return "aa"


@fixture(scope="module")
def folder_path():
    return "processed_data"


@fixture(scope="module")
def needed_encodings():
    return ["aa_seq", "aa_1hot"]


# ======== Tests ======


def test_hfds_from_pydict_aa(aa_expected_new_features):
    """hfds_from_pydict test: Create a dataset from a dictionary"""
    hfds = hfds_from_pydict(
        aa_example_0_missing_val,
        needed_encodings=aa_expected_new_features,
        as_DatasetDict=False,
    )
    assert all(col in list(hfds.features.keys()) for col in aa_expected_new_features)


def test_hfds_from_pydict_dna(dna_expected_new_features):
    """hfds_from_pydict test: Create a dataset from a dictionary"""
    hfds = hfds_from_pydict(
        dna_example_1_missing_val,
        needed_encodings=dna_expected_new_features,
        as_DatasetDict=False,
    )
    assert all(col in list(hfds.features.keys()) for col in dna_expected_new_features)


def test_save_dataset_as_csv(regr_dataset, regr_dataset_splits, needed_encodings):
    """save_dataset_as_csv test: Save a dataset as a csv file
    Args:
        regr_dataset (@fixture): Regression dataset
        needed_encodings (@fixture): list of encodings needed for regr_dataset
    """

    # filename as string
    save_dataset_as_csv(regr_dataset, "regression_toy_dataset_aa_seq_aa_1hot.csv")
    assert os.path.exists("regression_toy_dataset_aa_seq_aa_1hot.csv")
    os.remove("regression_toy_dataset_aa_seq_aa_1hot.csv")

    # filename as Path object
    save_dataset_as_csv(regr_dataset, Path("regression_toy_dataset_aa_seq_aa_1hot.csv"))
    assert os.path.exists("regression_toy_dataset_aa_seq_aa_1hot.csv")
    os.remove("regression_toy_dataset_aa_seq_aa_1hot.csv")

    # dataset as DatasetDict
    save_dataset_as_csv(
        regr_dataset_splits, "regression_toy_dataset_aa_seq_aa_1hot.csv"
    )
    assert os.path.exists("regression_toy_dataset_aa_seq_aa_1hot_test.csv")
    assert os.path.exists("regression_toy_dataset_aa_seq_aa_1hot_train.csv")
    os.remove("regression_toy_dataset_aa_seq_aa_1hot_train.csv")
    os.remove("regression_toy_dataset_aa_seq_aa_1hot_test.csv")


def test__read_csv_ignore_case(regr_dataset):
    """test for test__read_csv_ignore_case"""
    # basic use case
    save_dataset_as_csv(regr_dataset, "regression_toy_dataset_aa_seq_aa_1hot.csv")
    df = _read_csv_ignore_case("regression_toy_dataset_aa_seq_aa_1hot.csv")
    os.remove("regression_toy_dataset_aa_seq_aa_1hot.csv")
    assert df.shape == (21, 4)

    # missing file
    with pytest.raises(FileNotFoundError):
        _ = _read_csv_ignore_case("regression_toy_dataset_aa_seq_aa_1hot.csv")


def test_read_dataset_from_file():
    """testing read_dataset_from_file function"""
    pd.DataFrame(
        {
            "aa_mut": ["wildtype", "T8M", "P3G"],
            "aa_seq": ["GMPKSEFTHC", None, None],
            "target foo": [1, 2, 3],
        }
    ).to_csv("tempfile.csv", index=False)
    # returning a Dataset
    dataset = read_dataset_from_file("tempfile.csv")
    assert isinstance(dataset, Dataset)

    # test reading excel xlsx file
    # generate an excel file from the dataset we have here
    df = dataset.to_pandas()
    df.to_excel("tempfile.xlsx", index=False)
    ds = read_dataset_from_file("tempfile.xlsx")
    assert isinstance(ds, Dataset)

    # test reading excel xls file
    # generate an excel file from the dataset we have here
    df = ds.to_pandas()
    df.to_excel("tempfile.xls", index=False, engine="openpyxl")
    ds = read_dataset_from_file("tempfile.xls")
    assert isinstance(ds, Dataset)

    # Input a filename with capital letters
    pd.DataFrame(aa_example_0_no_missing_val).to_csv("tEsT_DATA.CsV", index=False)
    ds = read_dataset_from_file("tEsT_DATA.CsV")
    clean_up()
    assert all(
        feature in list(dataset.features.keys()) for feature in ["aa_seq", "target foo"]
    )

    # test reading file with wrong format
    df.to_html("tempfile.html", index=False)
    with pytest.raises(TypeError):
        df = read_dataset_from_file("tempfile.html")

    clean_up()


def test_get_missing_values():
    """get_missing_values test: Missing values in "aa_seq" column"""
    missing_val = get_missing_values(
        Dataset.from_dict(aa_example_0_missing_val), "aa_seq"
    )
    assert sum(missing_val) == 2


def test_fill_missing_sequences():
    """fill_missing_sequences test: Missing values in "aa_seq" column"""
    dataset_filled = fill_missing_sequences(
        Dataset.from_dict(aa_example_0_missing_val), "aa_seq"
    )
    assert dataset_filled["aa_seq"] == ["GMPKSEFTHC", "GMPKSEFMHC", "GMGKSEFTHC"]


def test_preprocess_data_aa(aa_expected_new_features):
    """preprocess_data  - aa"""
    # Missing values in "aa_seq" column
    pd.DataFrame(aa_example_0_missing_val).to_csv("tempfile.csv", index=False)
    dataset = preprocess_data("tempfile.csv")
    clean_up()
    assert all(
        feature in list(dataset.features.keys()) for feature in aa_expected_new_features
    )

    # No missing values in "aa_seq" column
    pd.DataFrame(aa_example_0_no_missing_val).to_csv("tempfile.csv", index=False)
    dataset = preprocess_data("tempfile.csv")
    clean_up()
    assert all(
        feature in list(dataset.features.keys()) for feature in aa_expected_new_features
    )


def test_preprocess_data_dna(dna_expected_new_features):
    """preprocess_data dna"""
    # Missing values in "dna_seq" column
    pd.DataFrame(dna_example_1_missing_val).to_csv("tempfile.csv", index=False)
    dataset = preprocess_data("tempfile.csv")
    clean_up()
    assert all(
        [
            feature in list(dataset.features.keys())
            for feature in dna_expected_new_features
        ]
    )

    # No missing values in "dna_seq" column
    pd.DataFrame(dna_example_1_no_missing_val).to_csv("tempfile.csv", index=False)
    dataset = preprocess_data("tempfile.csv")
    clean_up()
    assert all(
        [
            feature in list(dataset.features.keys())
            for feature in dna_expected_new_features
        ]
    )


def test_preprocess_data_invalid_alphabet():
    """preprocess_data test: Containing invalid AA alphabets"""
    pd.DataFrame(aa_example_0_invalid_alphabet).to_csv("tempfile.csv", index=False)
    with pytest.raises(ValueError):
        _ = preprocess_data("tempfile.csv")

    clean_up()


def test_preprocess_data_local_filesytem(aa_expected_new_features, folder_path):
    """preprocess_data test: using local file system"""
    pd.DataFrame(aa_example_0_no_missing_val).to_csv("tempfile.csv", index=False)
    local_filesystem = fsspec.filesystem("file")
    dataset = preprocess_data(
        "tempfile.csv", save_to_path=folder_path, filesystem=local_filesystem
    )
    clean_up()
    assert all(
        feature in list(dataset.features.keys()) for feature in aa_expected_new_features
    )


def test_load_similarity_dna(alphabet_type_dna):
    """Test load_similarity with DNA alphabet"""
    # Load single DNA similarity matrix and replace disk cache
    similarity_name = "Simple"
    _, similarity_matrix = load_similarity(
        alphabet_type_dna, similarity_name, replace_existing=True
    )
    assert similarity_matrix[0][0] == 1 and similarity_matrix[-1][-1] == 1

    # Load multiple DNA similarity matrix and replace disk cache
    similarity_name = ["Simple", "Identity"]
    _, similarity_matrix = load_similarity(
        alphabet_type_dna, similarity_name, replace_existing=True
    )
    assert similarity_matrix[0][0][-1] == 0 and similarity_matrix[1][2][2] == 1

    #  Without overwriting an existing DNA similarity matrix (replace_existing=False)
    similarity_name = "Simple"
    _, similarity_matrix = load_similarity(
        alphabet_type_dna, similarity_name, replace_existing=False
    )
    assert similarity_matrix[0][0] == 1 and similarity_matrix[-1][-1] == 1


def test_load_similarity_aa(alphabet_type_aa):
    """Test load_similarity with AA alphabet"""
    # Load single AA similarity matrix and replace disk cache
    similarity_name = "BLOSUM62"
    _, similarity_matrix = load_similarity(
        alphabet_type_aa, similarity_name, replace_existing=True
    )
    assert similarity_matrix[0][0] == 4 and similarity_matrix[13][4] == -3

    # Load multiple AA similarity matrix and replace disk cache
    similarity_name = ["BLOSUM62", "BLOSUM90", "IDENTITY", "PAM20"]
    _, similarity_matrix = load_similarity(
        alphabet_type_aa, similarity_name, replace_existing=True
    )
    assert (
        similarity_matrix[0][0][0] == 4
        and similarity_matrix[1][10][10] == 7
        and similarity_matrix[2].shape[0] == 21
        and similarity_matrix[3][0][4] == -9
    )

    # Without overwriting an existing AA similarity matrix (replace_existing=False)
    similarity_name = "PAM20"
    _, similarity_matrix = load_similarity(
        alphabet_type_aa, similarity_name, replace_existing=False
    )
    assert similarity_matrix[0][4] == -9 and similarity_matrix[-1][1] == -19


def test_load_similarity_invalid_alphabet():
    """Test load_similarity: Invalid alphabet type"""
    alphabet_type_bb = "bb"
    similarity_name = "Simple"
    with pytest.raises(ValueError):
        load_similarity(alphabet_type_bb, similarity_name)


if False:
    # ======== Tests to implement ======
    def test_load_similarity_dim_and_constency():
        """Raised error when the dimensions and consistency of the matrix is not as expected"""
        raise (NotImplementedError)

    def test_load_similarity_missing_entries():
        """Raised error for missing entries in the similarity matrix"""
        raise (NotImplementedError)
