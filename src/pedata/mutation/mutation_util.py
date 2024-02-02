from ..constants import Mut, Mut_with_no_targ


def convert_tuple_to_valid_namedtuple(t: tuple):
    """
    Converts a tuple to a valid namedtuple representation based on certain criteria.
    This function is designed to handle cases where other methods may return mutation named tuples with unexpected attributes.
    It allows you to create the appropriate namedtuple based on the length of the input tuple, ensuring code compatibility with other parts of the program.
    Only  mutation namedtuples with atleast two attributes: (pos src) and atmost three attributes (pos src targ) are considered valid.

    Args:
        t (tuple): The input tuple.

    Returns:
        namedtuple: The appropriate named tuple based on the length of the input tuple.

    Raises:
        ValueError: If the length of the input tuple is neither 2 nor 3.

    Example:

        >>> convert_tuple_to_valid_namedtuple((3, "A", "G"))
        Mut(pos=3, src='A', targ='G')

        >>> convert_tuple_to_valid_namedtuple((7, "K"))
        Mut(pos=7, src='K')

    """

    # Validate if input is a tuple
    if not isinstance(t, tuple):
        raise TypeError("Input only tuple mutation with at least 2 attributes")

    # Check the size of a tuple
    if len(t) == 3:
        return Mut(*t)
    elif len(t) == 2:
        return Mut_with_no_targ(*t)
    else:
        raise ValueError("Invalid tuple length. Expected length of 2 or 3.")
