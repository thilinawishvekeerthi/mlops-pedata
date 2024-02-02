import pytest
import pedata.mutation.mutation_util as mu
from pedata.constants import Mut, Mut_with_no_targ


class TestMutationUtil:
    def test_convert_tuple_to_valid_namedtuple(self):
        # Test case 1: With invalid input
        mut = 12
        with pytest.raises(TypeError):
            mu.convert_tuple_to_valid_namedtuple(mut)

        # Test case 2: Tuple with more than 3 attributes
        mut = (5, "K", "M", "J")
        with pytest.raises(ValueError):
            mu.convert_tuple_to_valid_namedtuple(mut)

        # Test case 3: Valid tuple
        mut = (3, "A", "G")
        assert mu.convert_tuple_to_valid_namedtuple(mut) == Mut(3, "A", "G")

        # Test case 5: Valid tuple with no target attribute
        mut = (7, "K")
        assert mu.convert_tuple_to_valid_namedtuple(mut) == Mut_with_no_targ(7, "K")
