import pedata.mutation.mutation_extractor as me
import pytest
from pedata.constants import Mut


def test_extract_mutation_namedtuples_from_sequences():
    #  Test case 1: Empty variant sequence
    parent_sequence = "ATCGATCG"
    variant_sequences = []
    with pytest.raises(ValueError):
        me.extract_mutation_namedtuples_from_sequences(
            variant_sequences, parent_sequence
        )


def test_extract_mutation_namedtuples_from_sequences2():
    #  Test case 2: Empty parent sequence
    parent_sequence = ""
    variant_sequences = ["ATCGATCG"]
    with pytest.raises(ValueError):
        me.extract_mutation_namedtuples_from_sequences(
            variant_sequences, parent_sequence
        )


def test_extract_mutation_namedtuples_from_sequences3():
    # Test case 3: Single variant sequence
    parent_sequence = "ATCGATCG"
    variant_sequences = ["ATCGTTCG"]
    assert me.extract_mutation_namedtuples_from_sequences(
        variant_sequences, parent_sequence
    ) == (
        [[Mut(pos=4, src="A", targ="T")]],  # List of changed mutations
        [],  # No invalid variant sequence found
    )


def test_extract_mutation_namedtuples_from_sequences_parent_in():
    # Test case 4: Multiple variant sequences
    parent_sequence = "ATCGATCG"
    variant_sequences = ["ATCGTTCG", "ATCGATCG", "ATTGATCG", "ATCGATCC"]
    extracted = me.extract_mutation_namedtuples_from_sequences(
        variant_sequences, parent_sequence
    )
    expected = (
        [
            [Mut(4, "A", "T")],
            [],  # No mutations, as this is the parent sequence
            [Mut(2, "C", "T")],
            [Mut(7, "G", "C")],
        ],  # List of mutations per variant
        [],  # No invalid variant sequence found
    )
    assert extracted == expected


def test_extract_mutation_namedtuples_from_sequences4():
    # Test case 4: Multiple variant sequences
    parent_sequence = "ATCGATCG"
    variant_sequences = ["ATCGTTCG", "ATTGATCG", "ATCGATCC"]
    assert me.extract_mutation_namedtuples_from_sequences(
        variant_sequences, parent_sequence
    ) == (
        [
            [Mut(4, "A", "T")],
            [Mut(2, "C", "T")],
            [Mut(7, "G", "C")],
        ],  # List of changed mutations
        [],  # No invalid variant sequence found
    )


def test_extract_mutation_namedtuples_from_sequences5():
    # Test case 5: Unequal length of sequences
    parent_sequence = "ATCGT"
    variant_sequences = ["ATCGTTCG", "ATCGATCG"]
    extracted = me.extract_mutation_namedtuples_from_sequences(
        variant_sequences, parent_sequence
    )
    expected = (
        [],  # No mutations
        ["ATCGTTCG", "ATCGATCG"],  # List of invalid variant sequences found
    )
    assert extracted == expected


def test_extract_mutation_namedtuples_from_sequences6():
    # Test case 6: Valid variant sequences with no changed mutations
    parent_sequence = "ATCGATCG"
    variant_sequences = ["ATCGATCG", "ATCGATCG", "ATCGATCG"]
    assert me.extract_mutation_namedtuples_from_sequences(
        variant_sequences, parent_sequence
    ) == (
        [[], [], []],  # No changed mutations
        [],  # No invalid variant sequence found
    )


def test_extract_mutation_namedtuples_from_sequences7():
    # Test case 7: Test on multiple mutations in one variant
    variant_sequences = ["AAAA", "ABBA"]
    parent_sequence = "BBAA"
    assert me.extract_mutation_namedtuples_from_sequences(
        variant_sequences, parent_sequence
    ) == (
        [
            [Mut(pos=0, src="B", targ="A"), Mut(pos=1, src="B", targ="A")],
            [Mut(pos=0, src="B", targ="A"), Mut(pos=2, src="A", targ="B")],
        ],
        [],  # No invalid variant sequence found
    )
