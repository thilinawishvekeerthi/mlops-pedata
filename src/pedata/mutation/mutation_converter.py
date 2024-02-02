""" The mutation_converter module contains functions for converting a mutation object type into another type."""
import numpy as np
from ..integrity import check_mutation_namedtuple
from ..constants import Mut
from .mutation import Mutation
from .mutation_util import convert_tuple_to_valid_namedtuple


def convert_variant_mutation_to_str(
    mut: list[Mut], delimiting_char: str = "_", offset: int = 0
) -> str:
    """
    Convert a list of mutations into a mutation code.

    Args:
        mut: Mutations in the format (position, source, target)
        delimiting_char: Character used to connect the mutations (default is "_")
        offset: Offset value to adjust mutation positions (default is 0)

    Returns:
        Mutation code encoded with delimiting character

    Examples:
            >>> mutations = [Mut(12, "A", "T"), Mut(23, "T", "G"), Mut(34, "C", "A")]
            >>> convert_variant_mutation_to_str(mutations)
            'A13T_T24G_C35A'
    """

    # Check if mut is a list
    if not isinstance(mut, list):
        raise TypeError(
            f"Invalid input: Expected a list of mutations but got {type(mut)}"
        )

    # Check if mut is not an empty list
    if len(mut) == 0:
        raise TypeError("Invalid input: Expected a non-empty list of mutations")

    # Verify mutations
    for m in mut:
        check_mutation_namedtuple(m)

    # Collect src & targ attributes from mutations
    tmp = []
    sources = [mutation.src for mutation in mut]
    targets = [mutation.targ for mutation in mut]

    # Convert mutation tuples to mutation namedtuples before sorting
    mut = [convert_tuple_to_valid_namedtuple(m) for m in mut]

    sorted_mutations = Mutation.sort_mutations_by_pos(mut)

    # Loop over sorted positions
    sorted_positions = [mutation[0] for mutation in sorted_mutations]
    for i, pos in enumerate(sorted_positions):
        tmp.append(
            sources[i] + str(pos - offset + 1) + targets[i]
        )  # Add the mutated base to the str

    return delimiting_char.join(tmp)


def convert_all_variant_mutations_to_str(
    mut: list[list[Mut]], delimiting_char: str = "_", offset: int = 0
) -> list:
    """Turn a sequence of mutation sequences into a list of string encodings.

    Args:
        mut: Sequence of mutation sequences, one for each variant.
        delimiting_char: Character used to connect the mutations for one variant (default is "_")
        offset: Offset value to adjust mutation positions (default is 0)

    Returns:
        Returns all mutations converted to string encodings only when the mutation has valid sequences of namedtuples, otherwise raise ValuesError

    Examples:
    >>> mutation = [
    ...    [Mut(12, "A", "T"), Mut(23, "T", "G")],
    ...    [Mut(15, "M", "K"), Mut(23, "G", "A")],
    ... ]
    >>> convert_all_variant_mutations_to_str(mutation)
    ['A13T_T24G', 'M16K_G24A']

    """

    # Check if mut is a list
    if not isinstance(mut, list) or len(mut) == 0:
        raise TypeError(
            f"Invalid input: Expected a list of mutation but got {type(mut)}"
        )

    str_encoding = []  # List to store strings of mutations

    # Check if mut contains lists of namedtuples
    for m in mut:
        if not isinstance(m, list):
            raise TypeError(
                f"Invalid input: The input list should also contain only lists of mutations but got {type(m)}"
            )

        # Validate mutation namedtuples
        for mut_tuple in m:
            check_mutation_namedtuple(mut_tuple)

        # Convert each mutation sequence to string
        encoded_mut = convert_variant_mutation_to_str(
            m, delimiting_char=delimiting_char, offset=offset
        )

        str_encoding.append(encoded_mut)  # Add the string encoding to the list

    return str_encoding


def dict_to_namedtuple_mut(mut: list[dict[str, list]]) -> list[Mut]:
    """
    Convert a list of dictionaries with 'position', 'source', and 'target' keys
    into a list of namedtuples 'Mut' with attributes 'pos', 'src', and 'targ'.

    Args:
        dicts: List of dictionaries containing 'position', 'source', and 'target' keys.

    Returns:
        List of namedtuples 'Mut'.

    Example:
        input_dicts = [{'source': ['A'], 'target': ['C'], 'position': [1]}, {'source': ['A'], 'target': ['D'], 'position': [1]}]
        resultant_namedtuples = dict_to_mut(input_dicts)

        Output:
        [ Mut(1, 'A', 'C'), Mut(1, 'A', 'D') ]

    """

    mut_list = []  # Stores a list of namedtuples

    # Extract namedtuples from input dictionary
    for d in mut:
        if isinstance(d, Mut):
            mut_list.append(d)
            continue

        pos = np.array(
            d["position"]
        ).item()  # Assuming 'position' contains a single element array
        src = d["source"][0]  # Assuming 'source' contains a single element array
        targ = d["target"][0]  # Assuming 'target' contains a single element array

        # Add extracted attributes to the list
        mut_list.append(Mut(pos, src, targ))

    return mut_list


def namedtuple_to_dict_mut(mut_list: list[Mut]) -> list[dict[str, list]]:
    """
    Convert a list of namedtuples 'Mut(pos, src, targ)' back into a Dictionary with 'position', 'source', and 'target' keys.

    Args:
        mut_list: List of namedtuples 'Mut'.

    Returns:
        Dictionary containing 'position', 'source', and 'target' keys.

    Example:
        dict_mut = namedtuple_to_dict_mut([ Mut(1, 'A', 'C'), Mut(2, 'A', 'D') ])

        Output:
        {'position': [1, 2], 'source': ['A', 'A'], 'target': ['C', 'D']}
    """

    dict_mut = {"position": [], "source": [], "target": []}

    # Extract data from namedtuples and create dictionaries
    for mut in mut_list:
        dict_mut["position"].append(mut.pos)
        dict_mut["source"].append(mut.src)
        dict_mut["target"].append(mut.targ)

    return dict_mut
