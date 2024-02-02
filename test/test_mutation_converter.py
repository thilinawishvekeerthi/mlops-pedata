import pytest
import pedata.mutation.mutation_converter as mc
from collections import namedtuple
from pedata.constants import Mut


class TestMutationConverter:
    def Test_convert_variant_mutation_to_str(self):
        # Test case 1: Non-list mutations
        mutation = (23, "K", "R")
        with pytest.raises(TypeError):
            mc.convert_variant_mutation_to_str(mutation)

        # Test case 2: Empty list of mutations
        mutations = []
        with pytest.raises(TypeError):
            mc.convert_variant_mutation_to_str(mutations)

        # # Test case 3: A list but without a valid sequence of tuples
        mutations = [{"position": 1, "source": "A", "target": "B"}]
        with pytest.raises(TypeError):
            mc.convert_variant_mutation_to_str(mutations)

        # Test case 4: Mutations with missing position and target
        mutations = namedtuple("mutation", "src")
        mutations = [mutations("K"), ("B")]
        with pytest.raises(TypeError):
            mc.convert_variant_mutation_to_str(mutations)

        # Test case 5: Mutations with a string position
        mutations = [Mut("24", "K", "B"), Mut("89", "B", "C")]
        with pytest.raises(TypeError):
            mc.convert_variant_mutation_to_str(mutations)

        # Test case 6: Mutations with a int source
        mutations = [Mut(24, 2, "K"), Mut(89, 3, "M")]
        with pytest.raises(TypeError):
            mc.convert_variant_mutation_to_str(mutations)

        # Test case 7: Mutations with a int target
        mutations = [Mut(24, "K", 2), Mut(89, "M", 3)]
        with pytest.raises(TypeError):
            mc.convert_variant_mutation_to_str(mutations)

        # Test case 8: Valid mutation with position, source, and target
        mutations = [Mut(12, "A", "T"), Mut(23, "T", "G"), Mut(34, "C", "A")]
        assert mc.convert_variant_mutation_to_str(mutations) == "A13T_T24G_C35A"

    def test_convert_all_variant_mutations_to_str(self):
        # Test case 1: Empty mutation list
        mutation = []
        with pytest.raises(TypeError):
            mc.convert_all_variant_mutations_to_str(mutation)

        # Test case 2: A non-list mutation type
        mutation = "Invalid input"
        with pytest.raises(TypeError):
            mc.convert_all_variant_mutations_to_str(mutation)

        # Test case 3: List of a single mutation namedtuple
        mutation = [Mut(70, "T", "A")]  # This should be a list os lists
        with pytest.raises(TypeError):
            mc.convert_all_variant_mutations_to_str(mutation)

        # Test case 4: List of mutations that contain non namedtuple mutations
        mutation = [[(12, "A", "T"), (23, "T", "G")]]
        with pytest.raises(TypeError):
            mc.convert_all_variant_mutations_to_str(mutation)

        # Test case 5: List of a single list with a single mutation namedtuple
        mutation = [[Mut(70, "T", "A")]]
        assert mc.convert_all_variant_mutations_to_str(mutation) == ["T71A"]

        # Test case 6: List of lists with a multiple mutations namedtuples
        mutation = [
            [Mut(12, "A", "T"), Mut(23, "T", "G")],
            [Mut(12, "A", "T"), Mut(23, "T", "G")],
        ]
        assert mc.convert_all_variant_mutations_to_str(mutation) == [
            "A13T_T24G",
            "A13T_T24G",
        ]
