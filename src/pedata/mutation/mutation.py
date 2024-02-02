""" The mutation module contains a Mutation class containing methods for applying mutations to sequences. 

The different methods allow to return the mutation saved as Named Tuple objects `Mut`.

The `Mut` objects are defined in `pedata.constants` module and contain information about the mutation (position, source, target).

"""
from typing import Callable
import re
import datasets as ds
import numpy as np
from collections import namedtuple
from copy import deepcopy
from ..integrity import check_dataset, check_mutation_namedtuple
from ..constants import Mut
from .mutation_util import convert_tuple_to_valid_namedtuple


class Mutation:
    @staticmethod
    def sort_mutations_by_pos(mut: list[Mut]):
        """Sort a list of mutations by position

        Args:
            mut List of mutations to be sorted. Each mutation should be a tuple with three elements: (position, source, target).

        Returns:
            Sorted list of mutations based on position.

        Raises:
            TypeError: If any mutation is invalid or does not match the expected format.

        Examples:
        >>> mutations = [Mut(2, "A", "C"),Mut(1, "G", "T"), Mut(3, "C", "G")]
        >>> Mutation.sort_mutations_by_pos(mutations)
        [Mut(pos=1, src='G', targ='T'), Mut(pos=2, src='A', targ='C'), Mut(pos=3, src='C', targ='G')]
        """

        # Check if mutations is a list
        if not isinstance(mut, list) or len(mut) == 0:
            raise TypeError("Input a non-empty list of mutations")

        # Verify mutations
        for m in mut:
            check_mutation_namedtuple(m)

        # Check if mutations are not already ordered
        if mut != sorted(mut, key=lambda x: x.pos):
            # Sort mutations by position
            sorted_mutations = sorted(mut, key=lambda x: x.pos)

            # Change tuple with no target to be Mut_with_no_tart(pos, src)
            for i, m in enumerate(sorted_mutations):
                sorted_mutations[i] = convert_tuple_to_valid_namedtuple(m)

            return sorted_mutations

        return mut  # Mutations are already ordered, return the original list

    @staticmethod
    def parse_variant_mutations(
        changes: str,
        valid: re.Pattern = re.compile(r"^[a-zA-Z]\s*[0-9]+\s*[a-zA-Z]$"),
        extractor: Callable = lambda c: (int(c[1:-1]), c[0], c[-1]),
        delimiting_char: str = "_",
        offset: int = 0,
    ) -> list[Mut]:
        """Parse each change/mutation in a list of changes.

        Args:
            changes: List of mutations, typically in the format SPT where
                S is the source amino acid or DNA base,
                P is the position of the mutation, and
                T is the target amino acid or DNA base.
                Generally speaking, the format is given by the `valid` and `extractor` variables.
            valid: Regexp to check if change is valid. Defaults to re.compile(r"^[a-zA-Z][0-9]+[a-zA-Z]$").
            extractor: Function to extract S, P, and T variables into a tuple.
                Defaults to lambda c: (int(c[1:-1]), c[0], c[-1]).
            delimiting_char: Delimiter character for splitting the changes. Defaults to "_".
            offset: Length of transport peptide i.e. offset for position/P variables. Defaults to 0.

        Returns:
            list[Mut]: List of mutation namedtuples (pos, src, targ)

        Example:
            >>> Mutation.parse_variant_mutations("G34C_L33T")
            [Mut(pos=32, src='L', targ='T'), Mut(pos=33, src='G', targ='C')]

        """

        mutations = []

        if "wildtype" in changes:
            return mutations

        # Split changes into individual mutations
        for c in changes.split(delimiting_char):
            c = c.strip()

            # Check if the mutation format is valid
            if valid.match(c) is None:
                raise ValueError(f"No valid mutation found in variant '{changes}'")

            # Extract position, source, and target using the provided extractor function
            p, s, t = extractor(c)
            mutations.append((int(p), str(s), str(t)))

        # Convert mutation tuples to mutation namedtuples before sorting
        mutations = [convert_tuple_to_valid_namedtuple(m) for m in mutations]

        # Sort mutations by position
        mutations = Mutation.sort_mutations_by_pos(mutations)

        # Adjust positions according to offset
        mutations = [(p - 1 + offset, s, t) for p, s, t in mutations]

        # Convert mutation tuples to mutation namedtuples again before returning
        mutations = [convert_tuple_to_valid_namedtuple(m) for m in mutations]

        return mutations

    @staticmethod
    def parse_all_mutations(
        dataset: ds.Dataset,
        offset: int = 0,
        valid: re.Pattern = re.compile(r"^[a-zA-Z]\s*[0-9]+\s*[a-zA-Z]$"),
        extractor: Callable = lambda c: (int(c[1:-1]), c[0], c[-1]),
        delimiting_char: str = "_",
    ) -> tuple[list[Mut], list[int]]:
        """
        Parse all mutations for all variants, where single mutations are delimited by the delimiting_character.

        Args:
            dataset: Data set containing all mutation codes in a `aa_mut` or `dna_mut` column.
            offset: Offset of positions in mutation codes.
                Defaults to 0.
            valid: Regular expression checking for the validity of a mutation string.
                Defaults to `re.compile(r"^[a-zA-Z]\s*[0-9]+\s*[a-zA-Z]$")`.
            extractor: Function for extracting a mutation triple containing `(position, source, target)`.
                Defaults to `lambda c: (int(c[1:-1]), c[0], c[-1])`.
            delimiting_char: Character that delimits individual mutations.
                Defaults to "_".

        Returns:
            (list[Mut], list[int]): A list of parsed mutations and a list of indexes representing non-parsed mutations.

        Example:
            >>> import datasets as ds
            >>> dataset = ds.Dataset.from_dict({"aa_mut": ["G34C_L33T", "K14M_A69G"], "aa_seq": [None, None], "target foo": [1, 2]})
            >>> parsed_mutations, non_parsed_idx = Mutation.parse_all_mutations(dataset)
            >>> print(parsed_mutations)
            [[Mut(pos=32, src='L', targ='T'), Mut(pos=33, src='G', targ='C')], [Mut(pos=13, src='K', targ='M'), Mut(pos=68, src='A', targ='G')]]
        """

        # Validate input dataset
        check_dataset(dataset)
        feature_keys = dataset.features.keys()

        if "aa_seq" in feature_keys:
            if "aa_mut" in feature_keys:
                mut_column = "aa_mut"

        if "dna_seq" in feature_keys:
            if "dna_mut" in feature_keys:
                mut_column = "dna_mut"

        parsed = []  # List to store parsed mutations
        non_parsed_idx = []  # List to store indexes representing non-parsed mutations

        # If "aa_mut" or "dna_mut" is missing, no parsing mutations needed
        if "mut_column" in locals():
            # Iterate over each amino acide or dna mutation
            for row_idx, var in enumerate(dataset[mut_column]):
                var = var.strip()

                # Check if the aa_mut or dna_mut is not wildtype or empty
                if var.lower() in ["wildtype (wt)", "wildtype", "", "wt", "-", "none"]:
                    non_parsed_idx.append(row_idx)
                    # Skip this row
                    continue

                # Parse the variant using the provided extractor function
                parsed.append(
                    Mutation.parse_variant_mutations(
                        var,
                        delimiting_char=delimiting_char,
                        offset=offset,
                        valid=valid,
                        extractor=extractor,
                    )
                )

        return parsed, non_parsed_idx

    @staticmethod
    def combine_variant_mutations(
        mut1: list[Mut], mut2: list[Mut], check_validity: bool = True
    ) -> list[Mut]:
        """
        Combine the mutations of one variant (represented as lists of Mut namedtuples) with those of another variant.

        Args:
            mut1: Mutations list representing all mutations in one variant.
            mut2: Mutations list representing all mutations in another variant.
            check_validity: Whether to check if `mut1` and `mut2` assume the same parent sequence.
                Defaults to True.

        Returns:
            List[Mut]: Combined mutations list.

        Example:
            >>> from pedata.constants import Mut
            >>> mut1 = [Mut(32, 'L', 'T'), Mut(33, 'G', 'C')]
            >>> mut2 = [Mut(13, 'K', 'M'), Mut(68, 'A', 'G')]
            >>> results = Mutation.combine_variant_mutations(mut1, mut2)
            >>> print(results) # doctest: +ELLIPSIS
            [Mut(pos=13, src='K', targ='M'), Mut(pos=32, src='L', targ='T'), ...]
        """

        # Validate mutations
        if not isinstance(mut1, list) or not isinstance(mut2, list):
            raise TypeError("Both mut1 and mut2 should be lists")

        if len(mut1) == 0 or len(mut2) == 0:
            raise ValueError("Both mut1 and mut2 should not be empty")

        # Validate mutation namedtuples in each list
        for mut_tuple in mut1:
            check_mutation_namedtuple(mut_tuple)
        for mut_tuple in mut2:
            check_mutation_namedtuple(mut_tuple)

        combined_mutations = []  # List to store the combined mutations

        # Combine mutations from mut1 and mut2
        for m1 in mut1:
            found_match = False
            for m2 in mut2:
                if m1.pos == m2.pos:
                    if check_validity and m1.src != m2.src:
                        raise ValueError(
                            "Combined mutations at the same position contain different source characters"
                        )

                    new_mut = deepcopy(m1)

                    if not check_validity and m1.src != m2.src:
                        for s in m2.src:
                            new_mut = new_mut._replace(src=new_mut.src + s)

                    for t in m2.targ:
                        combined_mutations.append(
                            new_mut._replace(targ=new_mut.targ + t)
                        )

                    found_match = True

            if not found_match:
                combined_mutations.append(m1)

        # Add mutations from mut2 that are not present in mut1
        for m2 in mut2:
            found_match = False
            for m1 in mut1:
                if m2.pos == m1.pos:
                    found_match = True
                    break
            if not found_match:
                combined_mutations.append(m2)

        # Sort combined mutations by position
        combined_mutations = Mutation.sort_mutations_by_pos(combined_mutations)

        return combined_mutations

    @staticmethod
    def generate_variant_mutation_combinations(
        variants1: list[Mut], variants2: list[Mut]
    ) -> list[Mut]:
        """
        Generate all possible combinations of mutations from two sets of variants.
            The function pairs each variant from variants1 with every variant from variants2. The resulting combined mutations are returned as a list of Mut namedtuples.
            This will allow a comprehensive analysis and comparison of different variant combinations.

        Args:
            variants1 (list[Mut]): List of mutations representing one set of variants.
            variants2 (list[Mut]): List of mutations representing another set of variants.

        Returns:
            list[Mut]: List of combined mutations representing all possible combinations.

        Example:
            >>> from pedata.constants import Mut
            >>> variants1 = [Mut(32, 'L', 'T'), Mut(33, 'G', 'C')]
            >>> variants2 = [Mut(13, 'K', 'M')]
            >>> results = Mutation.generate_variant_mutation_combinations(variants1, variants2)
            >>> print(results) # doctest: +NORMALIZE_WHITESPACE
            [[Mut(pos=13, src='K', targ='M'), Mut(pos=32, src='L', targ='T')],
            [Mut(pos=13, src='K', targ='M'), Mut(pos=33, src='G', targ='C')]]
        """

        # Check for valid lists
        if not isinstance(variants1, list) or not isinstance(variants2, list):
            raise TypeError(
                f"Invalid input: Expected two valid lists but got {type(variants1)} and {type(variants2)}"
            )

        # Check for non-empty lists
        if len(variants1) == 0 or len(variants2) == 0:
            raise ValueError(
                f"Invalid input: Expected non empty lists of variant mutations"
            )

        combined_mutations = []  # Stores all combined variant mutations

        # Generate combinations of mutations from variants1 and variants2
        for vm1 in variants1:
            for vm2 in variants2:
                combined_mutations.append(
                    Mutation.combine_variant_mutations([vm1], [vm2])
                )

        # Return combined mutations
        return combined_mutations

    @staticmethod
    def generate_variant_mutation_combinations_within_dataset(
        mut: list[Mut],
    ) -> list[Mut]:
        """Generate all possible combinations of mutations within the same dataset.

            This static method takes a list of mutations occurring in the dataset, where each variant mutation is represented as a named tuple 'Mut'.
            It combines the mutations from different variants in all possible ways and returns a list of lists representing the combined mutations.

        Args:
            mut (List[Mut]): A list of named tuples representing variant mutations in the dataset.
                            Each named tuple 'Mut' must have 'position', 'source', and 'target' attributes.

        Returns:
            List[List[Mut]]: A list of lists representing combined mutations.
                            Each inner list contains named tuples 'Mut' with 'position', 'source', and 'target' attributes.

        Examples:
            >>> from pedata.constants import Mut
            >>> mutations = [Mut(1, 'A', 'G'), Mut(2, 'C', 'T'), Mut(3, 'G', 'C'), Mut(4, 'T', 'A'), Mut(5, 'C', 'G')]
            >>> Mutation.generate_variant_mutation_combinations_within_dataset(mutations) # doctest: +NORMALIZE_WHITESPACE
            [[Mut(pos=1, src='A', targ='G'), Mut(pos=2, src='C', targ='T')],
            [Mut(pos=1, src='A', targ='G'), Mut(pos=3, src='G', targ='C')],
            [Mut(pos=1, src='A', targ='G'), Mut(pos=4, src='T', targ='A')],
            [Mut(pos=1, src='A', targ='G'), Mut(pos=5, src='C', targ='G')],
            [Mut(pos=2, src='C', targ='T'), Mut(pos=3, src='G', targ='C')],
            [Mut(pos=2, src='C', targ='T'), Mut(pos=4, src='T', targ='A')],
            [Mut(pos=2, src='C', targ='T'), Mut(pos=5, src='C', targ='G')],
            [Mut(pos=3, src='G', targ='C'), Mut(pos=4, src='T', targ='A')],
            [Mut(pos=3, src='G', targ='C'), Mut(pos=5, src='C', targ='G')],
            [Mut(pos=4, src='T', targ='A'), Mut(pos=5, src='C', targ='G')]]


        Note:
            The input mutations are expected to be valid and have the 'position', 'source', and 'target' attributes.
            The resulting combined mutations maintain the original order of positions for each variant mutation.
            The function uses combine_variant_mutations() function to check for the validity of the mutations and combines them.
        """

        # Check for valid list
        if not isinstance(mut, list):
            raise TypeError(f"Invalid input: Expected a valid list but got {type(mut)}")

        # Check for non-empty list
        if len(mut) == 0:
            raise ValueError(f"Invalid input: Expected non empty list of mutations")

        # If a single mutation is the input
        if len(mut) == 1:
            check_mutation_namedtuple(mut[0])
            return mut

        rval = []  # Initialize an empty list to store the resulting combined mutations

        # Iterate through each mutation in the dataset to form all possible combinations
        for i in range(len(mut)):
            for j in range(i + 1, len(mut)):
                # Combine the current two mutations using the combine_variant_mutations() method
                combined_mutations = Mutation.combine_variant_mutations(
                    [mut[i]], [mut[j]]
                )

                # Append the combined mutations to the result list
                rval.append(combined_mutations)

        return rval  # Return the final list containing all possible combinations

    @staticmethod
    def concat_mutations(mut: list[list[Mut]]) -> list[Mut]:
        """
        Combine multiple mutation lists into a single list of Mut namedtuples.

        Args:
            mut (List[List[Mut]]): A list of mutation lists, where each inner list contains Mut namedtuples.

        Returns:
            List[Mut]: A single list of Mut namedtuples combining all the mutation lists.

        Examples:
            >>> mut = [[(32, 'L', 'T'), (33, 'G', 'C')], [(13, 'K', 'M'), (68, 'A', 'G')]]
            >>> Mutation.concat_mutations(mut)
            [(32, 'L', 'T'), (33, 'G', 'C'), (13, 'K', 'M'), (68, 'A', 'G')]
        """

        rval = []  # List to store the combined mutations

        # Add mutation namedtuples to a single list
        for m in mut:
            if not isinstance(m, list):
                raise TypeError(
                    "Mutations should be a list of lists before concatinating"
                )
            rval.extend(m)

        return rval

    @staticmethod
    def get_parent_aa_seq(
        dataset: ds.Dataset, return_bool=False, return_idx=False
    ) -> str | np.ndarray | int:
        """
        Get the parent amino acid sequence from a dataset (identified by the variant name 'wildtype' or 'wt').
        By default returns the sequence as a string, but can also return the index of the parent sequence.

        Args:
            dataset: Dataset containing the parent sequence
            return_bool: Return a boolean array indicating the position(s) of the parent sequence in `dataset`. Defaults to False.
            return_idx: Return the index of the parent sequence. Defaults to False.

        Returns:
            Parent sequence as a string, boolean array, index integer

        Raises:
            TODO: write Raises

        Example:
            >>> data = {"aa_mut": ["wildtype","R227D", "K762M"], "aa_seq": ["ABCDEF", None, None], "target foo": [1, 2, 3]}
            >>> parent_seq = Mutation.get_parent_aa_seq(ds.Dataset.from_dict(data))
            >>> print(parent_seq)
            ABCDEF
        """

        # Validate dataset
        check_dataset(dataset)

        # Find all existig columns
        existing_columns = set(dataset.features.keys())

        if "aa_seq" in existing_columns:
            seq_column = "aa_seq"
            if "aa_mut" in existing_columns:
                mut_column = "aa_mut"

        if "dna_seq" in existing_columns:
            seq_column = "dna_seq"
            if "dna_mut" in existing_columns:
                mut_column = "dna_mut"

        # Check if they're any missing required columns
        if "mut_column" not in locals():
            raise Exception(
                f"Neither 'aa_mut' nor 'dna_mut' column found in dataset: {dataset}"
            )

        # Check if there is one "wildtype (WT)" value in "aa_mut" or "dna_mut" column
        wildtypes = [
            i for i, v in enumerate(dataset[mut_column]) if "wildtype" in v or "wt" in v
        ]
        if len(wildtypes) != 1:
            raise ValueError(
                f"Dataset contains {len(wildtypes)} instead of exactly one 'wildtype' entry in the '{mut_column}' column."
            )
        else:
            wildtype_idx = wildtypes[0]  # Store wildtype_idx as an int

        # Return as it was specified in arguments
        if return_bool:
            wildtype_bool = np.zeros_like(dataset[mut_column], dtype=bool)
            wildtype_bool[wildtype_idx] = True
            return wildtype_bool
        elif return_idx:
            return wildtype_idx
        else:
            return dataset[seq_column][wildtype_idx]

    @staticmethod
    def get_parent_sketch_from_mutations(
        mut: list[Mut], fill_character: str = "*"
    ) -> namedtuple("Sketch", "pos str"):
        """Sketch out the parent sequence from a list of mutations. This is useful, for example, to then estimate the offset of the position encoding.

        Args:
            mut: List of namedtuples representing mutations. Each namedtuple should contain at least the position (int) and source (str) respectively.
            fill_character: Character used to fill positions that do not occur in `mut`.
                Defaults to "*".

        Returns:
            Number of positions filled and resulting parent sketch string.

        Raises:
            ValueError: If `mut` is not a sequence of tuples.

        Examples:

            >>> from pedata.constants import Mut
            >>> mutations = [Mut(3, "A", "C"), Mut(2, "B", "D")]
            >>> Mutation.get_parent_sketch_from_mutations(mutations)
            (2, '**BA')
        """

        # Check if mutations is a list
        if not isinstance(mut, list) or len(mut) == 0:
            raise TypeError("Input a non-empty list of mutations")

        # Verify mutations
        for m in mut:
            check_mutation_namedtuple(m)

        # Extract the positions from the mutations
        positions = []
        for m in mut:
            positions.append(m.pos)

        # Find the maximum position
        max_pos = max(positions)

        # Calculate the number of unique positions in the mutations
        num_mut_pos = len(set(positions))

        # Create a list of fill characters with the length equal to the maximum position + 1
        parent = [fill_character] * (max_pos + 1)

        # Fill the parent sequence with the source characters at their respective positions
        multiple_source_characters = dict()
        for m in mut:
            if parent[m.pos] != fill_character and parent[m.pos] != m.src:
                # collect all source character conflicts
                if m.pos in multiple_source_characters:
                    # we already had conflicts – add the new source characters to the old
                    multiple_source_characters[m.pos] = multiple_source_characters[
                        m.pos
                    ].union({parent[m.pos], m.src})
                else:
                    # first conflict for this position
                    multiple_source_characters[m.pos] = {parent[m.pos], m.src}
            parent[m.pos] = m.src

        if len(multiple_source_characters) > 0:
            # sort the dictionary by key (position) and make a string for the error message
            multiple_source_characters = [
                f"pos {k}: {v}" for k, v in sorted(multiple_source_characters.items())
            ]
            raise ValueError(
                f"Multiple source characters found at the same position: {', '.join(multiple_source_characters)}"
            )

        # Join the parent list into a single string
        parent_sketch = "".join(parent)

        # Return the number of positions filled and the resulting parent sketch string
        return num_mut_pos, parent_sketch

    @staticmethod
    def estimate_offset(
        mut: list[Mut] | ds.Dataset,
        parent: str,
        delimiting_char: str = "_",
        most_likely: bool = False,
        only_perfect_matches: bool = True,
    ) -> ds.Dataset | int:
        """
        Return offsets in rank order of likelihood between a parent string and mutated sequences.

        This function estimates the offsets by trying out different offsets and computing the matching ratio
        between the parent sequence and the sketch of the parent sequence. The sketch is computed from the
        mutation codes provided. The offset with the highest matching ratio is considered the most likely offset.

        Args:
            mut: Mutations information represented as a namedtuple with atleast "source" and "position" attributes, or a DaDataset containing a "aa_mut" or "dna_mut" column. If a Dataset is provided, the `delimiting_char` is used to parse the mutations.
            parent: The parent string for which to find the offset.
            delimiting_char: The string delimiting individual mutations when parsing mutations from a Dataset.
                Defaults to "_".
            most_likely: If True, returns the most likely offset if the matching ratio is 100%; otherwise, raises an Exception.
                Defaults to False.
            only_perfect_matches: If True, only returns offsets with a matching ratio of 100%; otherwise, returns all offsets.

        Returns:
            If `most_likely` is False, returns a Dataset with the offsets in rank order of likelihood, including the matching ratio.
            If `most_likely` is True, returns the most likely offset as an integer.
            If the matching ratio is not 100% and `most_likely` is True, an Exception is raised.

        Raises:
            TODO: write Raises


        Examples:
            >>> import datasets as ds
            >>> parent_seq = "MEAPLSHV"
            >>> dataset = ds.Dataset.from_dict({"aa_mut": ["A3C_L5D", "S6K_M1A"], "aa_seq": [None, None], "target foo": [1, 2]})
            >>> result = Mutation.estimate_offset(dataset, parent=parent_seq)

        """

        input_mut = mut  # Stores the original dataset before further modifications

        # Handle Dataset input
        if isinstance(mut, ds.Dataset):
            # Find all existig columns
            existing_columns = set(mut.features.keys())
            if "aa_seq" in existing_columns:
                if "aa_mut" in existing_columns:
                    mut_column = "aa_mut"

            if "dna_seq" in existing_columns:
                if "dna_mut" in existing_columns:
                    mut_column = "dna_mut"

            # Validate dataset
            check_dataset(mut)
            parsed_mut, non_parsed_idx = Mutation.parse_all_mutations(
                mut, delimiting_char=delimiting_char
            )

            # Check if parsed mutation is empty
            if len(parsed_mut) == 0:
                raise ValueError(f"No parsed mutations found in dataset: \n{input_mut}")

            mut = Mutation.concat_mutations(parsed_mut)

        elif isinstance(mut, list) and all(isinstance(m, Mut) for m in mut):
            # Validate mutation namedtuples
            [check_mutation_namedtuple(m) for m in mut]

        else:
            raise TypeError("Input a valid list of mutation namedtuples or a dataset")

        # Initialize variables and arrays
        fill_character = " "
        num_mut, sketch = Mutation.get_parent_sketch_from_mutations(
            mut, fill_character=fill_character
        )
        sketch = sketch.rstrip()
        off = len(sketch) - len(sketch.strip())
        sketch = np.array(list(sketch.strip()), dtype=object)
        parent = np.array(list(parent), dtype=object).squeeze()
        not_fill = sketch != fill_character
        num_matching_pos = []
        non_matching_pos = []
        non_matching_src = []
        non_matching_par = []

        # Calculate the max_offset and raise an error is it's a negative number
        max_offset = len(parent) - len(sketch) + 1
        if max_offset < 0:
            raise Exception(
                f"The length of the parent sequence ({''.join(parent.astype(str))}) is smaller than the sketch derived from its mutations: {''.join(sketch.astype(str))}"
            )

        # Compare parent sequences with the sketch
        for i in range(max_offset):
            parent_window = parent[i : i + len(sketch)]
            match = (parent_window[not_fill] == sketch[not_fill]).squeeze()
            no_match = np.bitwise_not(match)
            num_matching_pos.append(match.sum())
            non_matching_pos.append(
                np.arange(off, off + len(sketch))[not_fill][no_match] + 1
            )
            non_matching_src.append(sketch[not_fill][no_match])
            non_matching_par.append(parent_window[not_fill][no_match])

        # Convert arrays to NumPy arrays and perform sorting
        num_matching_pos = np.atleast_1d(np.array(num_matching_pos).squeeze())
        non_matching_pos = np.atleast_1d(
            np.array(non_matching_pos, dtype=object).squeeze()
        )
        rank_order = np.argsort(num_matching_pos)[::-1].squeeze()
        rval = np.atleast_1d(
            (rank_order - off).squeeze()
        )  # Adjust the ranked order of offsets

        # Check if all mutations match the parent sequence
        if num_mut == num_matching_pos.max():
            rval = ds.Dataset.from_dict(
                {
                    "offset": rval,
                    "matching_ratio": np.atleast_1d(
                        num_matching_pos[rank_order] / num_mut
                    ),
                }
            )
        else:
            # Handle case when mutations don't match parent sequence
            rval = ds.Dataset.from_dict(
                {
                    "offset": [rval[0]],
                    "matching_ratio": [num_matching_pos[rank_order][0] / num_mut],
                    "non_match_pos": [str(non_matching_pos[rank_order[0]])],
                    "non_match_mutation_src": [str(non_matching_src[rank_order[0]])],
                    "non_match_parent_character": [
                        str(non_matching_par[rank_order[0]])
                    ],
                }
            )
        if only_perfect_matches or most_likely:
            if rval["matching_ratio"][0] == 1:
                rval = rval.filter(lambda x: x["matching_ratio"] == 1)
            else:
                raise ValueError(
                    f"Position values of mutation codes are inconsistent."
                    f"The most likely offset ({rval['offset'][0]}) has inconsistencies at positions {rval['non_match_pos'][0]}."
                )

        # Check if most_likely flag is set
        if not most_likely:
            return rval
        else:
            if len(rval) > 1:
                raise ValueError(
                    f"When processing mutation codes, multiple offsets seem to be plausible: {rval['offset']}. Cannot automaticcally decide for an offset."
                )
            return rval["offset"][0]

    @staticmethod
    def apply_variant_mutations(
        mutations: list[Mut], parent: str, offset: int = 0, check_validity: bool = False
    ) -> str:
        """
        Compute mutated sequences from parent sequence and mutations list.

        Args:
            mutations: List of namedtuples representing mutations.
                Each namedtuple should contain three attributes: (pos, src, targ).
            parent: Parent sequence to be mutated.
            offset: Offset of the mutation position to the parent sequence (default is 0).
            check_validity: True if validity of mutations is to be checked (default is False).

        Returns:
            Mutated sequence.

        Example:
            >>> from pedata.constants import Mut
            >>> mutations = [Mut(2, "A", "G"), Mut(4, "C", "T")]
            >>> parent_sequence = "TGAACC"
            >>> Mutation.apply_variant_mutations(
            ... mutations, parent_sequence, offset=1, check_validity=True
            ... )
            'TGAGCT'
        """

        if not isinstance(parent, str) or len(parent) == 0:
            raise TypeError("Parent sequence should be a string and not empty")

        seq = parent  # Initialize the mutated sequence as the parent sequence

        for m in mutations:
            # Calculate the position of the mutation by adding the offset
            pos = m.pos + offset

            # Raise an error if the source character at specific position in a mutation
            # does not match the character at the same position in the parent sequence
            if check_validity and m.src != seq[pos]:
                raise ValueError(
                    f"Mutation assumes '{m.src}' at position {pos} in parent, found '{seq[pos]}'"
                )

            # Apply the mutation: Replace the character at the mutation position in the sequence parent with the target character
            seq = seq[:pos] + m.targ + seq[pos + 1 :]

        return seq  # Return the mutated sequence

    @staticmethod
    def apply_all_mutations(
        mutations: ds.Dataset | list[list[Mut]],
        parent: str = None,
        offset: int = None,
        check_validity: bool = False,
        valid: re.Pattern = re.compile(r"^[a-zA-Z]\s*[0-9]+\s*[a-zA-Z]$"),
        extractor: Callable = lambda c: (int(c[1:-1]), c[0], c[-1]),
        delimiting_char: str = "_",
    ) -> list[str]:
        """
        Apply all mutations to a dataset and return the mutated sequences
        This method applies mutations to a dataset by iterating over each variant and applying the specified mutations.
        The resulting mutated sequences are returned as a list.

        Args:
            mutations: Dataset containing atleast an aa_mut or a dna_mut column.
            parent: Parent sequence.
                If not provided, it is assumed to be contained in the dataset,
                in the unique row where the 'variant' column contains the string 'wildtype'.
                Defaults to None.
            offset: Offset of the mutation position in relation to the parent sequence.
                If not provided, it is estimated.
                Defaults to None.
            check_validity: Whether to perform a validity check for the contained mutations.
                Defaults to False.
            valid: Regular expression to check the validity of a mutation string.
                Defaults to `re.compile(r"^[a-zA-Z]\s*[0-9]+\s*[a-zA-Z]$")`.
            extractor: Function to extract a mutation triple containing `(position, source, target)`.
                Defaults to `lambda c: (int(c[1:-1]), c[0], c[-1])`.
            delimiting_char: Delimiting character for the mutations.
                Defaults to "_".

        Returns:
            List of all mutated sequences.

        Raises:
            TypeError: If the input is not a Dataset.
            ValueError: If more than one variant without mutations is found.

        Example:
            >>> import datasets as ds
            >>> dataset = ds.Dataset.from_dict(
            ...     {
            ...         "aa_mut": ["wildtype", "C2X_E4Y", "F5T"],
            ...         "aa_seq": ["ABCDEFGH", None, None],
            ...         "target foo": [1, 2, 3]
            ...     }
            ... )
            >>> mutated_sequences = Mutation.apply_all_mutations(dataset)
            >>> print(mutated_sequences)
            ['ABCDEFGH', 'ABXDYFGH', 'ABCDETGH']
        """

        rval = []  # List to store the resulting mutated sequences.
        no_mutation = []  # Counter for the number of variants without mutations found.
        parsed_mutations = []
        # Check if the input is a Dataset
        if not isinstance(mutations, ds.Dataset) and not isinstance(mutations, list):
            raise TypeError("Input either a dataset or a list of namedtuple mutations")

        elif len(mutations) == 0:
            raise ValueError("The input dataset should not be empty")

        if isinstance(mutations, ds.Dataset):
            dataset = mutations

            # Parse mutations if the parsed_mutations column/feature is not present
            parsed_mutations, no_mutation = Mutation.parse_all_mutations(
                dataset,
                delimiting_char=delimiting_char,
                valid=valid,
                extractor=extractor,
            )

            # Raise an Exception if they're more more than one mutations that were not changed
            if len(no_mutation) > 1:
                raise Exception(
                    "More than one variant without mutations found. This is not supported."
                )

            # If no parsed mutations found
            if len(parsed_mutations) == 0:
                return rval

            # If parent is not provided, assume it is contained in the dataset
            if parent is None:
                parent = Mutation.get_parent_aa_seq(dataset)

        elif isinstance(mutations, list):
            parsed_mutations = mutations
            if parent is None:
                raise ValueError("Input a parent sequence")

        # If offset is not provided, estimate it based on the mutations
        if offset is None:
            offset = Mutation.estimate_offset(
                Mutation.concat_mutations(parsed_mutations), parent, most_likely=True
            )

        # Iterate over each variant and apply the mutations
        for row_idx in range(len(mutations)):
            # Append parent sequence where there is no mutations
            if row_idx in no_mutation:
                rval.append(parent)

            # Apply variant mutations to the parent sequence
            if row_idx < len(parsed_mutations):
                mut = parsed_mutations[row_idx]
                rval.append(
                    Mutation.apply_variant_mutations(
                        mut, parent, offset=offset, check_validity=check_validity
                    )
                )

        return rval
