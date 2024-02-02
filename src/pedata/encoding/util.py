from typing import Callable
from .base import EncodingSpec


def find_function_order(
    encoders: list[EncodingSpec],
    provided_encodings: list[str],
    required_encodings: list[str],
    satisfy_all: bool = True,
) -> list[Callable]:
    """
    Find the order in which to call encoding functions such that the required encodings can be computed.
    If the requirements are not satisfiable, throw an exception.

    Args:
        encoders (List[EncodingSpec]): List of encoding specifications.
        provided_encodings (List[str]): List of globally provided encodings.
        required_encodings (List[str]): List of required encodings.
        satisfy_all (bool, optional): If True, all encodings in `required_encodings` must be satisfied.
            If False, satisfy those that are satisfiable. Defaults to True.

    Returns:
        List[Callable]: A list of encoder functions in the order they should be called to satisfy the requirements.

    Raises:
        ValueError: If the requirements cannot be satisfied.

    Example:
        >>> from pedata.encoding import base
        >>> f1 = lambda x: x
        >>> f2 = lambda x: None
        >>> f3 = lambda x: 1
        >>> encoders = [
        ...     base.EncodingSpec(["aa_len"], ["aa_seq"], f1),
        ...     base.EncodingSpec(["aa_1hot"], ["aa_seq"], f2),
        ...     base.EncodingSpec(["dna_len"], ["dna_seq"], f3)
        ... ]
        >>> provided_encodings = ["aa_seq"]
        >>> required_encodings = ["aa_len"]
        >>> satisfy_all = True
        >>> result = find_function_order(encoders, provided_encodings, required_encodings, satisfy_all)
        >>> print(result) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [EncodingSpec(provides=['aa_len'], requires=['aa_seq'], ...)]
    """

    encoding_functions = {}  # Stores encoding functions for each encoder
    encodings_provided = {}  # Stores all avaialble encodings
    encodings_required = {}  # Stores all required/needed encodings
    encoding_specs = {}  # Stores all encoders

    if not satisfy_all:
        encodings = required_encodings
        required_encodings = []
        for encoding in encodings:
            # Skip encodings that require 'bnd_idcs' when it's not provided
            if (
                encoding.startswith("atm") or encoding.startswith("bnd")
            ) and "bnd_idcs" not in provided_encodings:
                continue

            # Skip encodings that require 'smiles_seq' when it's not provided
            if (
                encoding.startswith("smiles")
            ) and "smiles_seq" not in provided_encodings:
                continue

            if any(
                provided_enc.startswith("aa") for provided_enc in provided_encodings
            ):
                # If any Amino Acid encoding exists in provided_encodings, exclude DNA encodings
                if not encoding.startswith("dna"):
                    required_encodings.append(encoding)

            elif any(
                provided_enc.startswith("dna") for provided_enc in provided_encodings
            ):
                # If any DNA encoding exists in provided_encodings, exclude AA encodings
                if not encoding.startswith("aa"):
                    required_encodings.append(encoding)

    # Collect all necessary details from encoders
    for encoder in encoders:
        for encoding in encoder.provides:
            encoding_functions[encoding] = encoder.func
        encodings_provided[encoder.func] = encoder.provides
        encodings_required[encoder.func] = encoder.requires
        encoding_specs[encoder.func] = encoder

    # Topological sort using depth-first search
    sorted_list = []  # Store sorted list of functions
    checked = set()  # Stores encoding functions that have already been sorted

    def check_encoding(required_encodings: list[str]) -> None:
        """
        Check the providers of the required encodings.

        Args:
            required_encodings (List[str]): A list of required encodings.

        Raises:
            ValueError: If the requirements cannot be satisfied.
        """

        for encoding in required_encodings:
            # Check if encoding doesn't already exist
            if encoding not in provided_encodings:
                if satisfy_all and encoding not in encoding_functions:
                    raise ValueError(
                        f"Requirements cannot be satisfied. Lacking function for encoding: {encoding}"
                    )

                check_function(encoding_functions[encoding])

    def check_function(function: Callable):
        """
        Check if all function's encodings meet their requirements.

        Args:
            func (Callable): The function to check.
        """

        if function not in checked:
            checked.add(function)
            check_encoding(encodings_required[function])
            provided_encodings.extend(encodings_provided[function])
            sorted_list.append(function)

    check_encoding(required_encodings)

    if satisfy_all:
        # Check if all required encodings are in the sorted_list
        unsatisfied_requirements = set(required_encodings) - set(provided_encodings)

        # Throw an exception if there are unsatisfied requirements
        if unsatisfied_requirements:
            raise ValueError(
                f"Requirements cannot be satisfied: {unsatisfied_requirements}."
            )

    return [encoding_specs[f] for f in sorted_list]
