from pedata.config import add_encodings
import datasets as ds
from datasets import load_dataset

# import jax.numpy as np
import numpy as np
import pytest
from pytest import fixture
from pedata.disk_cache import read_dataset_from_file


def test_add_encodings_aa():
    """Add Amino Acid type encodings"""
    # Basics
    needed = ["aa_len", "aa_1gram", "aa_1hot"]
    dataset = ds.Dataset.from_dict(
        {
            "aa_mut": ["wildtype", "L2A"],
            "aa_seq": ["MLGTK", "MAGTK"],
            "target foo": [1, 2],
        }
    )
    encoded = add_encodings(dataset, needed)
    assert np.array(encoded["aa_1hot"][0]).sum() == len(dataset["aa_seq"][0])
    assert all(column in list(encoded.features.keys()) for column in needed)

    # Add aa_len encoding to a dataset dictionary.
    train_dataset = ds.Dataset.from_dict(
        {"aa_mut": ["wildtype", "L2A", "E5G"], "aa_seq": ["MLGKT", "MAGKT", "MEGKT"]}
    )
    test_dataset = ds.Dataset.from_dict(
        {"aa_mut": ["M3T", "P7S"], "aa_seq": ["MPGKT", "MFGKT"]}
    )
    dataset_dict = ds.DatasetDict({"train": train_dataset, "test": test_dataset})
    needed = ["aa_len", "aa_1hot"]
    encoded = add_encodings(dataset_dict, needed)
    for encoded_dataset in encoded.values():
        assert all(column in list(encoded_dataset.features.keys()) for column in needed)


def test_add_encodings_dna():
    """Add DNA type encodings"""
    # Basics
    dataset = ds.Dataset.from_dict(
        {
            "dna_mut": ["wildtype", "T2A"],
            "dna_seq": ["GTCTAG", "GACTAG"],
            "target foo": [1, 2],
        }
    )
    needed = ["dna_1hot", "dna_len"]
    encoded = add_encodings(dataset, needed)
    assert all(column in list(encoded.features.keys()) for column in needed)

    # List of encodings is empty
    needed = []
    dataset = ds.Dataset.from_dict(
        {
            "dna_mut": ["wildtype", "C3G", "A2T"],
            "dna_seq": ["GACTAG", "GAGTAG", "GTCTAG"],
            "bnd_idcs": [
                np.array([[1], [2]]),
                np.array([[1], [2]]),
                np.array([[1], [2]]),
            ],
            "target foo": [1, 2, 3],
        }
    )

    encoded = add_encodings(dataset.with_format("numpy"), needed)
    assert all(
        column in list(encoded.features.keys())
        for column in [
            "dna_mut",
            "dna_seq",
            "bnd_idcs",
            "target foo",
            "atm_count",
            "bnd_count",
            "atm_adj",
            "atm_bnd_incid",
            "dna_len",
            "dna_1hot",
        ]
    )


@fixture(scope="module")  # fixture for regr_dataset
def needed_encodings():
    return ["aa_len", "aa_1hot"]


if False:

    def test_add_encodings_atm_type_previous():
        # Test case 4: BACE Dataset
        dataset = load_dataset("Company/BACE", split="train")
        needed = ["atm_count"]
        encoded = add_encodings(dataset.with_format("numpy"), needed)
        assert all(column in list(encoded.features.keys()) for column in needed)


def test_add_encodings_atm_type():
    # Test case 4: BACE Dataset
    dataset = read_dataset_from_file(
        "src/pedata/static/example_data/small_mol_ds.parquet",
    )
    needed = ["atm_count"]
    encoded = add_encodings(dataset.with_format("numpy"), needed)
    assert all(column in list(encoded.features.keys()) for column in needed)


def test_add_encodings_smiles():
    """Add Amino Acid type encodings"""
    # Basics
    needed = ["smiles_1hot"]
    dataset = ds.Dataset.from_dict(
        {
            "smiles_seq": ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "O=S(=O)(O)CCS(=O)(=O)O"],
            "target foo": [1, 2],
        }
    )
    encoded = add_encodings(dataset, needed)
    assert all(column in list(encoded.features.keys()) for column in needed)
    assert len(dataset["smiles_seq"][0]) == np.array(encoded["smiles_1hot"][0]).sum()


def test_add_encodings_raises():
    # Test case 6: Datatype that is neither a dataset nor a dataset dictionary
    invalid_input = {"dna_mut": ["wildtype", "C3G", "A2T"], "target foo": [1, 2, 3]}
    with pytest.raises(TypeError):
        add_encodings(invalid_input, {"dna_len"})

    # Test case 7: Invalid encoding input
    dataset = ds.Dataset.from_dict(
        {
            "dna_mut": ["wildtype", "C3G", "A2T"],
            "dna_seq": ["GACT", "GAGT", "GTCT"],
            "target foo": [1, 2, 3],
        }
    )
    invalid_input = "invalid"
    with pytest.raises(TypeError):
        add_encodings(dataset, invalid_input)

    # Test case 8: Invalid AA Alphabests
    dataset = ds.Dataset.from_dict(
        {
            "aa_mut": ["wildtype", "C3B_T4O", "G1J_A2U"],
            "aa_seq": ["GACT", "GABO", "JUCT"],
            "target foo": [1, 2, 3],
        }
    )
    needed = ["aa_1hot"]
    with pytest.raises(ValueError):
        add_encodings(dataset, needed)

    # Test case 9: Invalid DNA Alphabests
    dataset = ds.Dataset.from_dict(
        {
            "dna_mut": ["wildtype", "C3K", "A2L"],
            "dna_seq": ["GACTAG", "GAKAG", "GLCTAG"],
            "target foo": [1, 2, 3],
        }
    )
    needed = ["dna_1hot"]
    with pytest.raises(ValueError):
        add_encodings(dataset, needed)

    # Test case 10: Lacking required column for encoding "dna_1hot"
    dataset = ds.Dataset.from_dict(
        {
            "aa_mut": ["wildtype", "C3G", "A2T"],
            "aa_seq": ["GACT", "GAGT", "GTCT"],
            "target foo": [1, 2, 3],
        }
    )
    needed = ["dna_1hot"]
    with pytest.raises(ValueError):
        add_encodings(dataset, needed)

    # Test case 11: Applying AA encodings on DNA dataset (DNA codes are converted to Amino Acids)
    dataset = ds.Dataset.from_dict(
        {
            "dna_mut": ["wildtype", "C3G", "A2T"],
            "dna_seq": ["GACCTA", "GAGCCA", "GTCGTC"],
        }
    )
    needed = ["aa_seq"]
    encoded = add_encodings(dataset, needed)
    assert all(
        column in list(encoded.features.keys())
        for column in ["aa_seq", "dna_mut", "dna_seq"]
    )

    # Test case 12: Applying AA encodings on DNA dataset along with other encodings that depend on aa_seq
    dataset = ds.Dataset.from_dict(
        {
            "dna_mut": ["wildtype", "C3G", "A2T"],
            "dna_seq": ["GACCTA", "GAGCCA", "GTCGTC"],
        }
    )
    needed = ["aa_seq", "aa_1hot", "aa_len"]
    encoded = add_encodings(dataset, needed)
    assert all(
        column in list(encoded.features.keys())
        for column in ["aa_seq", "aa_1hot", "aa_len", "dna_mut", "dna_seq"]
    )
