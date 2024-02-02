import datasets as ds
import pytest
from pedata.mutation.mutation import Mutation
from pedata.constants import Mut, Mut_with_no_targ
import re

from pedata import (
    aa_example_0_missing_target,
    aa_example_0_no_missing_val,
    dna_example_1_missing_target,
)


class TestMutation:
    def test_sort_mutations_by_pos(self):
        # Test case 1: Empty list of mutations
        mutations = []
        with pytest.raises(TypeError):
            Mutation.sort_mutations_by_pos(mutations)

        # Test case 2: List of mutations with valid format and unordered positions
        mutations = [Mut(2, "A", "C"), Mut(1, "G", "T"), Mut(3, "C", "G")]
        assert Mutation.sort_mutations_by_pos(mutations) == [
            (1, "G", "T"),
            (2, "A", "C"),
            (3, "C", "G"),
        ]

        # Test case 3: List of mutations with valid format and ordered positions
        mutations = [Mut(1, "G", "T"), Mut(2, "A", "C"), Mut(3, "C", "G")]
        assert Mutation.sort_mutations_by_pos(mutations) == [
            (1, "G", "T"),
            (2, "A", "C"),
            (3, "C", "G"),
        ]

        # Test case 4: List with a mutations that has only position and source attributes
        mutations = [
            Mut(1, "G", "T"),
            Mut_with_no_targ(7, "A"),  # Missing target
            Mut(0, "C", "G"),
        ]
        assert Mutation.sort_mutations_by_pos(mutations) == [
            Mut(pos=0, src="C", targ="G"),
            Mut(pos=1, src="G", targ="T"),
            Mut_with_no_targ(pos=7, src="A"),
        ]

        # Test case 5: List of mutations with invalid format (invalid types)
        mutations = [
            Mut(1, "G", "T"),
            Mut("2", "A", "C"),  # Invalid position type (string instead of int)
            Mut(3, "C", "G"),
        ]
        with pytest.raises(TypeError):
            Mutation.sort_mutations_by_pos(mutations)

        # Test case 6: List of mutations with invalid format (source should be string)
        mutations = [
            Mut(1, "G", "T"),
            Mut(2, 8, "C"),  # Invalid source type (int instead of str)
            Mut(3, "C", "G"),
        ]
        with pytest.raises(TypeError):
            Mutation.sort_mutations_by_pos(mutations)

        # Test case 7: List of mutations with invalid format (target should be string)
        mutations = [
            Mut(1, "G", "T"),
            Mut(2, "K", 8),  # Invalid source type (int instead of str)
            Mut(3, "C", "G"),
        ]
        with pytest.raises(TypeError):
            Mutation.sort_mutations_by_pos(mutations)

    def test_parse_variant_mutations(self):
        # Test case 1: Invalid mutation format
        changes = "G_34_C"
        with pytest.raises(ValueError):
            Mutation.parse_variant_mutations(changes)

        # Test case 2: Single mutation
        changes = "G34C"
        assert Mutation.parse_variant_mutations(changes) == [Mut(33, "G", "C")]

        # Test case 3: Multiple mutations with offset
        changes = "G34C_L33T_A12G"
        assert Mutation.parse_variant_mutations(changes) == [
            Mut(11, "A", "G"),
            Mut(32, "L", "T"),
            Mut(33, "G", "C"),
        ]

        # Test case 4: No mutations (wildtype)
        changes = "wildtype"
        assert Mutation.parse_variant_mutations(changes) == []

    def test_parse_all_mutations(self):
        # Test case 1: Empty Dataset
        dataset = ds.Dataset.from_dict({})
        with pytest.raises(KeyError):
            Mutation.parse_all_mutations(dataset)

        # Test case 2: Dataset without a "aa_mut" column
        dataset = ds.Dataset.from_dict({"aa_seq": [None], "target foo": [1]})
        with pytest.raises(KeyError):
            Mutation.parse_all_mutations(dataset)

        # Test case 3: Dataset without a "dna_mut" column
        dataset = ds.Dataset.from_dict({"dna_seq": [None], "target foo": [1]})
        with pytest.raises(KeyError):
            Mutation.parse_all_mutations(dataset)

        # Test case 4: Missing column "aa_mut" but with no missing values in aa_seq
        dataset = ds.Dataset.from_dict(
            {"aa_seq": ["A3C_L5D", "S6K_M1A"], "target foo": [1, 2]}
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(dataset)
        assert parsed_mutations == [] and non_parsed_idx == []

        # Test case 5: Missing column "dna_mut" but with no missing values in dna_seq
        dataset = ds.Dataset.from_dict(
            {"dna_seq": ["A3C_L5D", "S6K_M1A"], "target foo": [1, 2]}
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(dataset)
        assert parsed_mutations == [] and non_parsed_idx == []

        # Test case 6: aa_mut column with invalid value
        dataset = ds.Dataset.from_dict(
            {
                "aa_mut": ["A3C_L5D", "none", "-"],
                "aa_seq": [None, None, None],
                "target foo": [1, 2, 3],
            }
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(dataset)
        assert parsed_mutations == [
            [Mut(2, "A", "C"), Mut(4, "L", "D")]
        ] and non_parsed_idx == [1, 2]

        # Test case 7: dna_mut column with invalid value
        dataset = ds.Dataset.from_dict(
            {
                "dna_mut": ["A3C_L5D", "none", "-"],
                "dna_seq": [None, None, None],
                "target foo": [1, 2, 3],
            }
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(dataset)
        assert parsed_mutations == [
            [Mut(2, "A", "C"), Mut(4, "L", "D")]
        ] and non_parsed_idx == [1, 2]

        # Test case 8: Single mutation parsing
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["G34C_L33T"], "aa_seq": [None], "target foo": [1]}
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(dataset)
        assert (
            parsed_mutations == [[(32, "L", "T"), (33, "G", "C")]]
            and non_parsed_idx == []
        )

        # Test case 9: Multiple mutations
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["G34C_L33T_A12G"], "aa_seq": [None], "target foo": [1]}
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(dataset)
        assert (
            parsed_mutations == [[(11, "A", "G"), (32, "L", "T"), (33, "G", "C")]]
            and non_parsed_idx == []
        )

        # Test case 10: No mutations (wildtype)
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["wildtype"], "aa_seq": ["MEAPLSHV"], "target foo": [1]}
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(dataset)
        assert parsed_mutations == [] and non_parsed_idx == [0]

        # Test case 11: Custom delimiting character
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["G34C;L33T"], "aa_seq": [None], "target foo": [1]}
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(
            dataset, delimiting_char=";"
        )
        assert (
            parsed_mutations == [[(32, "L", "T"), (33, "G", "C")]]
            and non_parsed_idx == []
        )

        # Test case 12: Custom offset
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["G34C_L33T"], "aa_seq": [None], "target foo": [1]}
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(
            dataset, offset=1
        )
        assert (
            parsed_mutations == [[(33, "L", "T"), (34, "G", "C")]]
            and non_parsed_idx == []
        )

        # Test case 13: No valid mutation found in aa_mut
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["G34_C"], "aa_seq": [None], "target foo": [1]}
        )
        with pytest.raises(ValueError):
            Mutation.parse_all_mutations(dataset, valid=re.compile(r"^[a-zA-Z][0-9]+$"))

        # Test case 14: No valid mutation found in dna_mut
        dataset = ds.Dataset.from_dict(
            {"dna_mut": ["G34_C"], "dna_seq": [None], "target foo": [1]}
        )
        with pytest.raises(ValueError):
            Mutation.parse_all_mutations(dataset, valid=re.compile(r"^[a-zA-Z][0-9]+$"))

        # Test case 15: Dataset with multiple aa_mut values
        dataset = ds.Dataset.from_dict(
            {
                "aa_mut": ["G34C_L33T_K35M", "M36A_G37C"],
                "aa_seq": [None, None],
                "target foo": [1, 2],
            }
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(dataset)
        assert (
            parsed_mutations
            == [
                [(32, "L", "T"), Mut(33, "G", "C"), Mut(34, "K", "M")],
                [Mut(35, "M", "A"), Mut(36, "G", "C")],
            ]
            and non_parsed_idx == []
        )

        # Test case 16: Dataset with multiple dna_mut values
        dataset = ds.Dataset.from_dict(
            {
                "dna_mut": ["G34C_L33T_K35M", "M36A_G37C"],
                "dna_seq": [None, None],
                "target foo": [1, 2],
            }
        )
        parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(dataset)
        assert (
            parsed_mutations
            == [
                [(32, "L", "T"), Mut(33, "G", "C"), Mut(34, "K", "M")],
                [Mut(35, "M", "A"), Mut(36, "G", "C")],
            ]
            and non_parsed_idx == []
        )

    def test_combine_variant_mutations(self):
        # Test case 1: mut1 is not a list
        mut1 = Mut(1, "A", "K")
        mut2 = [Mut(2, "T", "M")]
        with pytest.raises(TypeError):
            Mutation.combine_variant_mutations(mut1, mut2)

        # Test case 2: mut2 is not a list
        mut1 = [Mut(1, "A", "K")]
        mut2 = Mut(2, "T", "M")
        with pytest.raises(TypeError):
            Mutation.combine_variant_mutations(mut1, mut2)

        # Test case 3: With empty mut1
        mut1 = []
        mut2 = [Mut(2, "T", "M")]
        with pytest.raises(ValueError):
            Mutation.combine_variant_mutations(mut1, mut2)

        # Test case 4: With empty mut2
        mut1 = [Mut(1, "A", "K")]
        mut2 = []
        with pytest.raises(ValueError):
            Mutation.combine_variant_mutations(mut1, mut2)

        # Test case 5: mut1 contains an invalid namedtuple
        mut1 = [
            Mut("1", "A", "K"),
            Mut(2, "M", "K"),
        ]  # Position is a string in the first namedtuple
        mut2 = [Mut(3, "T", "A"), Mut(4, "K", "G")]
        with pytest.raises(TypeError):
            Mutation.combine_variant_mutations(mut1, mut2)

        # Test case 6: mut2 contains an invalid namedtuple
        mut1 = [Mut(1, "A", "K"), Mut(2, "M", "K")]
        mut2 = [
            Mut(3, "T", "A"),
            (4, "K", "G"),
        ]  # Second namedtuple is not an instance of Mut namedtuple
        with pytest.raises(TypeError):
            Mutation.combine_variant_mutations(mut1, mut2)

        # Test case 7: Mutations with different parent sequences (At the same position, the two mutation namedtuples contain different source characters)
        mut1 = [Mut(32, "L", "T")]
        mut2 = [Mut(32, "A", "G")]
        with pytest.raises(ValueError):
            Mutation.combine_variant_mutations(mut1, mut2)

        # Test case 8: No mutation namedtuples with similar positions
        mut1 = [Mut(32, "L", "T"), Mut(33, "G", "C")]
        mut2 = [Mut(13, "K", "M"), Mut(68, "A", "G")]
        results = Mutation.combine_variant_mutations(mut1, mut2)
        assert results == [
            Mut(13, "K", "M"),
            Mut(32, "L", "T"),
            Mut(33, "G", "C"),
            Mut(68, "A", "G"),
        ]

        # Test case 9: Mutations of different positions with check_validity set to False
        mut1 = [Mut(32, "L", "T"), Mut(33, "G", "C")]
        mut2 = [Mut(13, "K", "M"), Mut(68, "A", "G")]
        results = Mutation.combine_variant_mutations(mut1, mut2, check_validity=False)
        assert results == [
            Mut(13, "K", "M"),
            Mut(32, "L", "T"),
            Mut(33, "G", "C"),
            Mut(68, "A", "G"),
        ]

    def test_generate_variant_mutation_combinations(self):
        # Test Case 1: Empty variants lists
        variants1 = []
        variants2 = []
        with pytest.raises(ValueError):
            Mutation.generate_variant_mutation_combinations(variants1, variants2)

        # Test Case 2: Invalid input
        variants1 = {Mut(1, "A", "G")}
        variants2 = {Mut(2, "K", "T")}
        with pytest.raises(TypeError):
            Mutation.generate_variant_mutation_combinations(variants1, variants2)

        # Test Case 3: Single mutation in variants1 and variants2
        variants1 = [Mut(1, "A", "G")]
        variants2 = [Mut(2, "C", "T")]
        result = Mutation.generate_variant_mutation_combinations(variants1, variants2)
        print("\n\n ", result)
        assert (
            len(result) == 1
        ), f"Test Case 3 failed: Expected length 1 but got: {len(result)}"
        assert result[0] == [Mut(1, "A", "G"), Mut(2, "C", "T")], "Test Case 3 failed"

        # Test Case 4: Multiple mutations in variants1 and variants2
        variants1 = [Mut(1, "A", "G"), Mut(2, "C", "T")]
        variants2 = [Mut(3, "G", "C"), Mut(4, "T", "A")]
        result = Mutation.generate_variant_mutation_combinations(variants1, variants2)
        assert (
            len(result) == 4
        ), f"Test Case 4 failed: Expected length 4 but got: {len(result)}"
        assert result[0] == [Mut(1, "A", "G"), Mut(3, "G", "C")], "Test Case 4 failed"
        assert result[1] == [Mut(1, "A", "G"), Mut(4, "T", "A")], "Test Case 4 failed"
        assert result[2] == [Mut(2, "C", "T"), Mut(3, "G", "C")], "Test Case 4 failed"
        assert result[3] == [Mut(2, "C", "T"), Mut(4, "T", "A")], "Test Case 4 failed"

        # Test Case 5: Different lengths of variants1 and variants2
        variants1 = [Mut(1, "A", "G"), Mut(2, "C", "T")]
        variants2 = [Mut(3, "G", "C"), Mut(4, "T", "A"), Mut(5, "C", "G")]
        result = Mutation.generate_variant_mutation_combinations(variants1, variants2)
        assert (
            len(result) == 6
        ), f"Test Case 5 failed: Expected length 6 but got: {len(result)}"
        assert result[0] == [Mut(1, "A", "G"), Mut(3, "G", "C")], "Test Case 5 failed"
        assert result[1] == [Mut(1, "A", "G"), Mut(4, "T", "A")], "Test Case 5 failed"
        assert result[2] == [Mut(1, "A", "G"), Mut(5, "C", "G")], "Test Case 5 failed"
        assert result[3] == [Mut(2, "C", "T"), Mut(3, "G", "C")], "Test Case 5 failed"
        assert result[4] == [Mut(2, "C", "T"), Mut(4, "T", "A")], "Test Case 5 failed"
        assert result[5] == [Mut(2, "C", "T"), Mut(5, "C", "G")], "Test Case 5 failed"

    def test_generate_variant_mutation_combinations_within_dataset(self):
        # Test Case 1: Empty list
        mut = []
        with pytest.raises(ValueError):
            Mutation.generate_variant_mutation_combinations_within_dataset(mut)

        # Test Case 2: Invalid input
        mut = {Mut(13, "G", "M")}
        with pytest.raises(TypeError):
            Mutation.generate_variant_mutation_combinations_within_dataset(mut)

        # Test Case 3: Non-namedtuple mutations in the database
        mut = [(2, "G", "M"), (3, "K", "T"), (4, "A", "M")]
        with pytest.raises(TypeError):
            Mutation.generate_variant_mutation_combinations_within_dataset(mut)

        # Test Case 4: Single mutation in the database
        mut = [Mut(13, "G", "M")]
        combined_mutations = (
            Mutation.generate_variant_mutation_combinations_within_dataset(mut)
        )
        assert combined_mutations == mut

        # Test Case 5: Mutiple mutations in the database
        mut = [Mut(1, "A", "G"), Mut(2, "C", "T"), Mut(3, "G", "C"), Mut(4, "T", "A")]
        result = Mutation.generate_variant_mutation_combinations_within_dataset(mut)
        assert result[0] == [Mut(1, "A", "G"), Mut(2, "C", "T")], "Test Case 5 failed"
        assert result[1] == [Mut(1, "A", "G"), Mut(3, "G", "C")], "Test Case 5 failed"
        assert result[2] == [Mut(1, "A", "G"), Mut(4, "T", "A")], "Test Case 5 failed"
        assert result[3] == [Mut(2, "C", "T"), Mut(3, "G", "C")], "Test Case 5 failed"
        assert result[4] == [Mut(2, "C", "T"), Mut(4, "T", "A")], "Test Case 5 failed"
        assert result[5] == [Mut(3, "G", "C"), Mut(4, "T", "A")], "Test Case 5 failed"

    def test_concat_mutations(self):
        # Test case 1: with empty list
        mut = [[]]
        assert Mutation.concat_mutations(mut) == []

        # Test case 2: with multiple empty inner lists
        mut = [[], [], []]
        assert Mutation.concat_mutations(mut) == []

        # Test case 3: with a single list of mutations
        mut = [[(32, "L", "T"), (33, "G", "C")]]
        assert Mutation.concat_mutations(mut) == [(32, "L", "T"), (33, "G", "C")]

        # Test case 4: with multiple lists of mutations
        mut = [[(32, "L", "T"), (33, "G", "C")], [(13, "K", "M"), (68, "A", "G")]]
        assert Mutation.concat_mutations(mut) == [
            (32, "L", "T"),
            (33, "G", "C"),
            (13, "K", "M"),
            (68, "A", "G"),
        ]

        # Test case 5: with a mix of empty and non-empty lists
        mut = [
            [],
            [(32, "L", "T"), (33, "G", "C")],
            [],
            [(13, "K", "M"), (68, "A", "G")],
            [],
        ]
        assert Mutation.concat_mutations(mut) == [
            (32, "L", "T"),
            (33, "G", "C"),
            (13, "K", "M"),
            (68, "A", "G"),
        ]

    def test_get_parent_aa_seq(self):
        # Test case 1: Input dictionary
        dict = {
            "aa_mut": ["wildtype", "R227D", "K696M"],
            "aa_seq": ["ABCDEF", None, None],
            "target foo": [1, 2],
        }
        with pytest.raises(TypeError):
            Mutation.get_parent_aa_seq(dict)

        # Test case 2: Missing required columns
        empty_dataset = ds.Dataset.from_dict(
            {"aa_mut": [], "target foo": []}
        )  # Missing "aa_seq" column
        with pytest.raises(KeyError):
            Mutation.get_parent_aa_seq(empty_dataset)

        # Test case 3: Missing required "aa_mut" column to extract the parent sequence
        empty_dataset = ds.Dataset.from_dict(
            {"aa_seq": ["MAETK", "GMAFT"], "target foo": [1, 2]}
        )
        with pytest.raises(Exception):
            Mutation.get_parent_aa_seq(empty_dataset)

        # Test case 4: Multiple "wildtype (WT)" value in "aa_mut" column
        dataset = ds.Dataset.from_dict(
            {  # 2 widtype entries make this dictionary invalid
                "aa_mut": ["wildtype", "R227D", "wt"],
                "aa_seq": ["ABCDEF", None, "HIKLMN"],
                "target foo": [1, 2, 3],
            }
        )
        with pytest.raises(ValueError):
            Mutation.get_parent_aa_seq(dataset)

        # Below, the function is not expected to raise any errors
        dataset = ds.Dataset.from_dict(aa_example_0_no_missing_val)

        # Test case 5: Return_idx is true
        assert Mutation.get_parent_aa_seq(dataset, return_idx=True) == 0

        # Test case 6: Return_bool is true
        counter = 0
        for a in list(Mutation.get_parent_aa_seq(dataset, return_bool=True)):
            if a:
                counter += 1
        assert counter == 1

        # Test case 7: Both return_idx and return_bool are false
        assert ds.Dataset.from_dict(aa_example_0_no_missing_val)["aa_seq"][0]

    def test_get_parent_sketch_from_mutations(self):
        # Test case 1: Non-list mutations
        mutation = (23, "K")
        with pytest.raises(TypeError):
            Mutation.get_parent_sketch_from_mutations(mutation)

        # Test case 2: Empty list of mutations
        mutations = []
        with pytest.raises(TypeError):
            Mutation.get_parent_sketch_from_mutations(mutations)

        # # Test case 3: A list with no valid sequence of namedtuples
        mutations = [{"position": 1, "source": "A"}, {"position": 2, "source": "B"}]
        with pytest.raises(TypeError):
            Mutation.get_parent_sketch_from_mutations(mutations)

        # Test case 4: Mutations with missing position or source
        mutations = [("K"), ("B")]
        with pytest.raises(TypeError):
            Mutation.get_parent_sketch_from_mutations(mutations)

        # Test case 5: Mutations with a string position
        mutations = [Mut_with_no_targ("24", "K"), Mut_with_no_targ("89", "B")]
        with pytest.raises(TypeError):
            Mutation.get_parent_sketch_from_mutations(mutations)

        # Test case 6: Mutations with a int source
        mutations = [Mut_with_no_targ(24, 2), Mut_with_no_targ(89, 3)]
        with pytest.raises(TypeError):
            Mutation.get_parent_sketch_from_mutations(mutations)

        # Test case 8: Valid mutation with only position and source
        mutations = [Mut_with_no_targ(3, "A"), Mut_with_no_targ(2, "B")]
        expected_output = (2, "**BA")
        assert Mutation.get_parent_sketch_from_mutations(mutations) == expected_output

        # Test case 9: Valid mutation with position, source, and target
        mutations = [Mut(3, "A", "C"), Mut(2, "B", "D")]
        expected_output = (2, "**BA")
        assert Mutation.get_parent_sketch_from_mutations(mutations) == expected_output

        # Test case 10: raise an error if multiple mutation source characters have the same position
        with pytest.raises(ValueError):
            Mutation.get_parent_sketch_from_mutations(
                [
                    elem
                    for code in ["M1A", "B1C"]
                    for elem in Mutation.parse_variant_mutations(code)
                ]
            )

    def test_estimate_offset(self):
        # Test case 1: Dataset with missing column "aa_seq"
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["A3C_L5D", "S6K_M1A"], "target foo": [1, 2]}
        )
        parent_seq = "MEAPLSHV"
        with pytest.raises(KeyError):
            Mutation.estimate_offset(dataset, parent_seq)

        # Test case 2: Missing column "aa_mut"
        dataset = ds.Dataset.from_dict(
            {"aa_seq": ["A3C_L5D", "S6K_M1A"], "target foo": [1, 2]}
        )
        parent_seq = "MEAPLSHV"
        with pytest.raises(ValueError):
            Mutation.estimate_offset(dataset, parent_seq)

        # Test case 3: Parent sequence matches the sketch 100%
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["C2X_E4Y"], "aa_seq": [None], "target foo": [1]}
        )
        parent_seq = "ABCDEFGH"
        assert Mutation.estimate_offset(dataset, parent=parent_seq)["offset"][0] == 1

        # Test case 4: Parent sequence does not match the sketch
        parent_seq = "MEAPLSHV"
        dataset = ds.Dataset.from_dict(
            {
                "aa_mut": ["A3C_L4D", "S6K_M1A"],
                "aa_seq": [None, None],
                "target foo": [1, 2],
            }
        )
        # with only_perfect_matches=True this raises an exception
        with pytest.raises(Exception):
            Mutation.estimate_offset(dataset, parent=parent_seq, only_perfect_matches=True)["offset"][0] == 0
        
        # with most_likely=True this raises an exception
        with pytest.raises(Exception):
            Mutation.estimate_offset(dataset, parent=parent_seq, most_likely=True)

        # Test case 5: DNA dataset missing column starting with target
        dataset = ds.Dataset.from_dict(dna_example_1_missing_target)
        dataset_with_no_parent = dataset.select(range(1, len(dataset)))
        parent_seq = dna_example_1_missing_target["dna_seq"][0]
        with pytest.raises(KeyError):
            Mutation.estimate_offset(dataset_with_no_parent, parent_seq)["offset"][
                0
            ]  # Returns -70

        # Test case 6: AA dataset Missing column starting with target
        dataset = ds.Dataset.from_dict(aa_example_0_missing_target)
        parent_seq = aa_example_0_missing_target["aa_seq"][0]
        with pytest.raises(KeyError):
            Mutation.estimate_offset(dataset, parent=parent_seq)["offset"][
                0
            ]  # Returns -55

        # Test case 7: Input as Mutation namedtuple
        parent_seq = "ABCDEFGH"
        mutations = [Mut(2, "C", "X"), Mut(4, "E", "Y")]
        assert Mutation.estimate_offset(mutations, parent=parent_seq)["offset"][0] == 0

        # Test case 8: Most likely offset requested
        parent_seq = "ABCDEFGH"
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["C2X_E4Y"], "aa_seq": [None], "target foo": [1]}
        )
        assert (
            Mutation.estimate_offset(dataset, parent=parent_seq, most_likely=True) == 1
        )

        # Test case 9: Even if only one offset is tested, code still works
        parent_seq = "MB"
        dataset = ds.Dataset.from_dict(
            {
                "aa_mut": ["M1A", "B2C", "M1A_B2C"],
                "aa_seq": [None] * 3,
                "target foo": [1] * 3,
            }
        )

        assert len(Mutation.estimate_offset(dataset, parent=parent_seq)) == 1

    def test_estimate_offset_ambiguous(self):
        # Test case where the offset cannot be estimated automatically
        #parent_seq = "MEAPLSHV"
        parent_seq = "AAAAAAAA"
        dataset = ds.Dataset.from_dict(
            {
                "aa_mut": ["A3C_A4D", "A6K_A1A"],
                "aa_seq": [None, None],
                "target foo": [1, 2],
            }
        )
        
        # with most_likely=True this raises an exception
        with pytest.raises(Exception):
            Mutation.estimate_offset(dataset, parent=parent_seq, most_likely=True)

    def test_apply_variant_mutations(self):
        # Test case 1: No mutations
        mutations = []
        parent_sequence = "ATCG"
        assert Mutation.apply_variant_mutations(mutations, parent_sequence) == "ATCG"

        # Test case 1: No parent sequence
        mutations = [
            Mut(1, "A", "G"),
        ]
        parent_sequence = ""
        with pytest.raises(TypeError):
            Mutation.apply_variant_mutations(mutations, parent_sequence)

        # Test case 2: Single mutation
        mutations = [Mut(1, "T", "G")]
        parent_sequence = "ATCG"
        assert Mutation.apply_variant_mutations(mutations, parent_sequence) == "AGCG"

        # Test case 3: Multiple mutations
        mutations = [Mut(1, "A", "G"), Mut(3, "C", "T")]
        parent_sequence = "ATCG"
        assert Mutation.apply_variant_mutations(mutations, parent_sequence) == "AGCT"

        # Test case 4: Offset is set to 1 and check validity is true.
        # (Every source character in mutation is found in the parent sequence at the specified posistions)
        mutations = [Mut(2, "A", "G"), Mut(4, "C", "T")]
        parent_sequence = "TGAACC"
        assert (
            Mutation.apply_variant_mutations(
                mutations, parent_sequence, offset=1, check_validity=True
            )
            == "TGAGCT"
        )

        # Test case 4: Check validity is true.
        # (Some source characters in mutation are not found in the parent sequence at the specified posistions)
        mutations = [Mut(1, "A", "G"), Mut(2, "C", "T")]
        parent_sequence = (
            "ATCT"  # 'A' is not found in parent sequence at position 1, found 'T'
        )
        with pytest.raises(ValueError):
            Mutation.apply_variant_mutations(
                mutations, parent_sequence, check_validity=True
            )

    def test_apply_all_mutations(self):
        # Test case 1: With an invalid input type
        invalid_input = {"aa_mut": ["DAMDIW"], "aa_seq": [None], "target foo": [1]}
        with pytest.raises(TypeError):
            Mutation.apply_all_mutations(invalid_input)

        # Test case 2: When an empty Dataset is passed
        empty_dataset = ds.Dataset.from_dict(
            {"aa_mut": [], "aa_seq": [], "target foo": []}
        )
        with pytest.raises(ValueError):
            Mutation.apply_all_mutations(empty_dataset)

        # Test case 3: A valid dataset
        dataset = ds.Dataset.from_dict(
            {
                "aa_mut": ["wildtype", "C2X_E4Y", "F5T"],
                "aa_seq": ["ABCDEFGH", None, None],
                "target foo": [1, 2, 3],
            }
        )
        mutated_sequences = Mutation.apply_all_mutations(dataset)
        assert mutated_sequences == ["ABCDEFGH", "ABXDYFGH", "ABCDETGH"]

        # Test case 4: No parsed mutations found
        dataset = ds.Dataset.from_dict(
            {"aa_seq": ["METAFGH", "AKHMEWT"], "target foo": [1, 2]}
        )
        assert Mutation.apply_all_mutations(dataset) == []

        # Test case 5: Dataset with wildtype in aa_mut and no provided parent sequence in parameters
        dataset = ds.Dataset.from_dict(
            {
                "aa_mut": ["wildtype", "K3M", "A2T"],
                "aa_seq": ["MAKPS", None, None],
                "target foo": [1, 2, 3],
            }
        )
        Mutation.apply_all_mutations(dataset) == ["MAMPS", "MTKPS"]

        # Test case 6: Dataset with wildtype in dna_mut and no provided parent sequence in parameters
        dataset = ds.Dataset.from_dict(
            {
                "dna_mut": ["wildtype", "K3M", "A2T"],
                "dna_seq": ["MAKPS", None, None],
                "target foo": [1, 2, 3],
            }
        )
        Mutation.apply_all_mutations(dataset) == ["MAMPS", "MTKPS"]

        # Test case 7: More than one amino acide mutations have no mutations
        dataset = ds.Dataset.from_dict(
            {
                "aa_mut": ["wildtype", "K3M", "A2T", "-"],
                "aa_seq": ["MAKPS", None, None, None],
                "target foo": [1, 2, 3, 4],
            }
        )
        with pytest.raises(Exception):
            Mutation.apply_all_mutations(dataset)

        # Test case 8: More than one DNA mutations have no mutations
        dataset = ds.Dataset.from_dict(
            {
                "dna_mut": ["wildtype", "K3M", "A2T", "-"],
                "dna_seq": ["MAKPS", None, None, None],
                "target foo": [1, 2, 3, 4],
            }
        )
        with pytest.raises(Exception):
            Mutation.apply_all_mutations(dataset)

        # Test case 9: Dataset with no wildtype in aa_mut and no parent provided
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["K3M", "A2T"], "aa_seq": [None, None], "target foo": [1, 2]}
        )
        with pytest.raises(ValueError):
            Mutation.apply_all_mutations(dataset) == ["MAMPS", "MTKPS"]

        # Test case 10: Dataset with no wildtype in dna_mut and no parent provided
        dataset = ds.Dataset.from_dict(
            {"dna_mut": ["K3M", "A2T"], "dna_seq": [None, None], "target foo": [1, 2]}
        )
        with pytest.raises(ValueError):
            Mutation.apply_all_mutations(dataset) == ["MAMPS", "MTKPS"]

        # Test case 11: Both parent and offset are provided
        dataset = ds.Dataset.from_dict(
            {"aa_mut": ["T8M", "P3G"], "aa_seq": [None, None], "target foo": [1, 2]}
        )
        parent = "GMPKSEFTHBCX"
        offset = 0
        Mutation.apply_all_mutations(dataset, parent=parent, offset=offset) == [
            "GMPKSEFMHBCX",
            "GMGKSEFTHBCX",
        ]


TestMutation().test_generate_variant_mutation_combinations()
