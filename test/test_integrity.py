import pedata.integrity as integrity
import datasets as ds
import pytest


def test_check_dataset():
    # Test case 1: Invalid mutation
    invalid_mut = {"aa_mut": ["MLDSWE"], "aa_seq": [None]}  # Dictionary
    with pytest.raises(TypeError):
        integrity.check_dataset(invalid_mut)

    # Test case 2: Empty dataset
    empty_dataset = ds.Dataset.from_dict({"aa_mut": [], "aa_seq": [], "target": []})
    assert integrity.check_dataset(empty_dataset) is None

    # Test case 3: Missing "aa_seq" column
    invalid_dataset = ds.Dataset.from_dict({"aa_mut": [], "target": []})
    with pytest.raises(KeyError):
        integrity.check_dataset(invalid_dataset)

    # Test case 4: Missing "dna_seq" column
    invalid_dataset = ds.Dataset.from_dict({"dna_mut": [], "target": []})
    with pytest.raises(KeyError):
        integrity.check_dataset(invalid_dataset)

    # Test case 5: Missing "aa_mut" column
    invalid_dataset = ds.Dataset.from_dict(
        {"aa_seq": [None, None], "target foo": [1, 2]}
    )
    with pytest.raises(KeyError):
        integrity.check_dataset(invalid_dataset)

    # Test case 6: Missing "dna_mut" column
    invalid_dataset = ds.Dataset.from_dict(
        {"dna_seq": [None, None], "target foo": [1, 2]}
    )
    with pytest.raises(KeyError):
        integrity.check_dataset(invalid_dataset)

    # Test case 7: Missing a column starting with keyword "target"
    invalid_dataset = ds.Dataset.from_dict(
        {"dna_mut": ["AMKTG", "MTGK"], "dna_seq": [None, None]}
    )
    with pytest.raises(KeyError):
        integrity.check_dataset(invalid_dataset)

    # Test case 8: "target summary variable" column is present
    invalid_dataset = ds.Dataset.from_dict(
        {"dna_seq": ["MLGLYITR", "MAGLYITR"], "target summary variable": [1, 2]}
    )
    with pytest.raises(KeyError):
        integrity.check_dataset(invalid_dataset)

    # Test case 9: "aa_seq" has no missing values
    valid_dataset = ds.Dataset.from_dict(
        {"aa_seq": ["MLGLYITR", "MAGLYITR"], "target foo": [1, 2]}
    )
    assert integrity.check_dataset(valid_dataset) == None

    # Test case 10: "dna_seq" has no missing values
    valid_dataset = ds.Dataset.from_dict(
        {"dna_seq": ["MLGLYITR", "MAGLYITR"], "target foo": [1, 2]}
    )
    assert integrity.check_dataset(valid_dataset) == None

    # Test case 11: "aa_mut" column has missing values
    valid_dataset = ds.Dataset.from_dict(
        {
            "aa_seq": ["MLGLYITR", None],
            "aa_mut": ["wildtype", "L2R"],
            "target foo": [1, 2],
        }
    )
    assert integrity.check_dataset(valid_dataset) == None

    # Test case 12: "dna_mut" column has missing values
    valid_dataset = ds.Dataset.from_dict(
        {
            "dna_seq": ["MLGLYITR", None],
            "dna_mut": ["wildtype", "L2R"],
            "target foo": [1, 2],
        }
    )
    assert integrity.check_dataset(valid_dataset) == None

    # Test case 13: Missing column starting with keyword "target"
    invalid_dataset = ds.Dataset.from_dict(
        {
            "dna_seq": ["MLGLYITR", None],
            "dna_mut": ["wildtype", "L2R"],
        }
    )
    with pytest.raises(KeyError):
        integrity.check_dataset(invalid_dataset)
