import datasets as ds
import pytest
from Bio.Data import CodonTable
from pedata.encoding import translate_dna_to_aa_seq


def test_translate_dna_to_aa_seq():
    # Test case 1: invalid input
    invalid_input = {"dna_seq": ["GATCTG"]}
    with pytest.raises(TypeError):
        translate_dna_to_aa_seq(invalid_input)

    # Test case 2: Dataset missing 'dna_seq' column
    dataset = ds.Dataset.from_dict({"aa_seq": ["MAPTG", "GATKM"], "target foo": [1, 2]})
    with pytest.raises(TypeError):
        translate_dna_to_aa_seq(invalid_input)

    # Test case 3: Dataset with missing values in 'dna_seq' columns
    dataset = ds.Dataset.from_dict({"dna_seq": [None, None], "target foo": [1, 2]})
    with pytest.raises(ValueError):
        translate_dna_to_aa_seq(dataset)

    # Test case 4: Invalid DNA alphabets
    dataset = ds.Dataset.from_dict({"dna_seq": ["MAPKTS"]})
    with pytest.raises(CodonTable.TranslationError):
        translate_dna_to_aa_seq(dataset)

    # Test case 5: Translate a single dna_seq
    dataset = ds.Dataset.from_dict({"dna_seq": ["GATCTG"]})
    result = translate_dna_to_aa_seq(dataset)
    expected_output = {"aa_seq": ["DL"]}
    assert result["aa_seq"] == expected_output["aa_seq"]

    # Test case 6: Translate multiple dna_seq
    dataset = ds.Dataset.from_dict(
        {"dna_seq": ["GATCTG", "CTT", "AAGATTACACCT", "TAGGGACAAAATGCATTG"]}
    )
    result = translate_dna_to_aa_seq(dataset)
    expected_output = {"aa_seq": ["DL", "L", "KITP", "GQNAL"]}
    assert result["aa_seq"] == expected_output["aa_seq"]
