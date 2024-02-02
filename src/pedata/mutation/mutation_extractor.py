""" The mutation_extractor module contains functions for extracting mutations from sequences."""
from typing import Iterable
from ..constants import Mut
from .mutation_util import convert_tuple_to_valid_namedtuple
from .mutation_converter import convert_all_variant_mutations_to_str


def extract_mutation_namedtuples_from_sequences(
    variant_seq_str: list[str], parent: str, offset: int = 0
) -> tuple[list[Mut], list[str]]:
    """
    Extract mutations from sequences with respect to a parent sequence.

    Args:
        variant_seq_str: List of variant sequence strings.
        parent: Parent sequence.
        offset: Offset of the parent sequence and the interesting part (default is 0).

    Returns:
        A tuple containing a list of all mutation namedtuples (pos, src, targ),
        and a list of invalid variant sequences found (variant sequences with different lengths from the parent sequence).

    Example:
        >>> parent_sequence = "ATCGATCG"
        >>> variant_sequences = ["ATCGTTCG", "ATCGATCG", "ATTGATCG"]
        >>> result = extract_mutation_namedtuples_from_sequences(variant_sequences, parent_sequence)
        >>> print(result) # doctest: +NORMALIZE_WHITESPACE
        ([[Mut(pos=4, src='A', targ='T')],
        [Mut(pos=2, src='C', targ='T')]], [])
    """

    # Validate input
    if not isinstance(variant_seq_str, Iterable):
        raise TypeError(
            f"Invalid variant sequence string: Expected variant sequence to be list of strings but got {type(variant_seq_str)}"
        )

    if len(variant_seq_str) == 0:
        raise ValueError(
            "Invalid variant sequence string: Variant sequence should be a non-empty list of strings."
        )

    if not isinstance(parent, str):
        raise TypeError(
            f"Invalid parent: Expected parent to be a string but got {type(parent)}"
        )

    if len(parent) == 0:
        raise ValueError("Invalid parent: Parent should be a non-empty string.")

    if not isinstance(parent, str) or len(parent) == 0:
        raise TypeError("Parent sequence should be a string and not empty")

    changed_mut = []  # Store resulting list of mutation sequences that were changed
    invalid_var_seq = (
        []
    )  # List to track variant sequences with unequal lengths (where mutations were not changed)

    for var_idx, var in enumerate(variant_seq_str):
        mutation = []
        # Check for invalid variant sequences
        if len(var) != len(parent):
            invalid_var_seq.append(var)
            continue
        elif var != parent:
            # if the variant doesn't equal the parent sequence, compute the mutations
            # otherwise, the `mutation` list will stay empty
            for pos, source in enumerate(parent):
                if source != var[pos]:
                    mutation.append((pos - offset, source, var[pos]))

            # Store only mutations that were changed and convert them into valid namedtuples
            for mut_idx, mut_tuple in enumerate(mutation):
                if len(mut_tuple) > 1:
                    mutation[mut_idx] = convert_tuple_to_valid_namedtuple(mut_tuple)

            if len(mutation) == 0:
                raise Exception(
                    f"Invalid variant sequence: Variant sequence is not equal to parent sequence but no mutations were found.\nVariant: {var}\nParent:  {parent}"
                )

        changed_mut.append(mutation)

    return changed_mut, invalid_var_seq


def extract_mutation_str_from_sequences(
    variant_seq_str: Iterable[str],
    parent: str,
    offset: int = 0,
    delimiting_char: str = "_",
) -> list[str]:
    """Convert a list of sequences to a list of mutations.

        This function takes a list of variant sequence strings and a parent sequence string, and converts each variant sequence into a mutation code with respect to the parent sequence.
        The mutation code represents the changes in the variant sequence compared to the parent sequence.

    Args:
        variant_seq_str: A list of sequence strings representing variant sequences. Each sequence can be an explicit full amino acid (AA) or DNA sequence.
        parent: The parent sequence string relative to which the mutation codes are computed.
        offset: The offset of the mutation position if the first position in the sequence is not 1. Defaults to 0.
        delimiting_char: The delimiting character used to combine multiple mutations into a single code. Defaults to "_".

    Returns:
        A list of mutation codes, where each code corresponds to a variant sequence in the input.

    Example:
        >>> variant_sequences = ["AAAA", "ABBA"]
        >>> parent_sequence = "BBAA"
        >>> extract_mutation_str_from_sequences(variant_sequences, parent_sequence)
        ['B1A_B2A', 'B1A_A3B']

    Note:
        The function internally uses the `extract_mutation_namedtuples_from_sequences` function from the `me` module.
        The `extract_mutation_namedtuples_from_sequences` function takes the variant sequences, parent sequence, and offset as inputs, and returns the mutation objects.
        The `convert_all_variant_mutations_to_str` function is then used to convert the mutation objects into mutation codes.

    """

    # Extract all valid mutations from input sequences
    mutations, invalid_var_seq = extract_mutation_namedtuples_from_sequences(
        variant_seq_str, parent, offset
    )

    # Convert mutations to string
    encoded_mut = convert_all_variant_mutations_to_str(mutations, delimiting_char)

    # Return list of encoded mutations
    return encoded_mut
