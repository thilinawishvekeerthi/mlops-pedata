from pedata.encoding.util import find_function_order
import pytest


class EncodingSpec:
    def __init__(self, provides, requires, func):
        self.provides = provides
        self.requires = requires
        self.func = func


class TestEncodingUtil:
    def test_find_function_order(self):
        # Temporary encoding functions for tests
        f1 = lambda x: x
        f2 = lambda x: None
        f3 = lambda x: 1
        f4 = lambda x: x
        f5 = lambda x: x
        f6 = lambda x: x

        encoders = [
            EncodingSpec(["aa_len"], ["aa_seq"], f1),
            EncodingSpec(["aa_1hot"], ["aa_seq"], f2),
            EncodingSpec(["dna_len"], ["dna_seq"], f3),
            EncodingSpec(["dna_1hot"], ["dna_seq"], f4),
            EncodingSpec(["atm_count", "bnd_count"], ["bnd_idcs"], f5),
            EncodingSpec(["atm_bnd_incid"], ["atm_count", "bnd_count"], f6),
        ]

        # Test case 1: All required encodings are satisfiable
        provided_encodings = ["aa_seq"]
        required_encodings = ["aa_len"]
        satisfy_all = True
        expected_order = [f1]
        result = find_function_order(
            encoders, provided_encodings, required_encodings, satisfy_all
        )
        assert [
            encoder.func for encoder in result
        ] == expected_order, (
            "Test case 1 failed! Order of encoding functions is incorrect"
        )

        # Test case 2: Satisfy all is False with AA sequences in provided encodings
        provided_encodings = ["aa_seq"]
        required_encodings = [p for e in encoders for p in e.provides]
        satisfy_all = False
        expected_order = [f1, f2]
        result = find_function_order(
            encoders, provided_encodings, required_encodings, satisfy_all
        )
        assert [
            encoder.func for encoder in result
        ] == expected_order, (
            "Test case 2 failed! Order of encoding functions is incorrect"
        )

        # Test case 3: Satisfy all is False with DNA sequences in provided encodings
        provided_encodings = ["dna_seq"]
        required_encodings = [p for e in encoders for p in e.provides]
        satisfy_all = False
        expected_order = [f3, f4]
        result = find_function_order(
            encoders, provided_encodings, required_encodings, satisfy_all
        )
        assert [
            encoder.func for encoder in result
        ] == expected_order, (
            "Test case 3 failed! Order of encoding functions is incorrect"
        )

        # Test case 4: "bnd_idcs" is in provided encodings and Satisfy all is False
        provided_encodings = ["dna_seq", "bnd_idcs"]
        required_encodings = [p for e in encoders for p in e.provides]
        satisfy_all = False
        expected_order = [f3, f4, f5, f6]
        result = find_function_order(
            encoders, provided_encodings, required_encodings, satisfy_all
        )
        assert [
            encoder.func for encoder in result
        ] == expected_order, (
            "Test case 4 failed! Order of encoding functions is incorrect"
        )

        # Test case 5: No required encodings
        provided_encodings = ["aa_seq"]
        required_encodings = []  # Empty List
        expected_order = []
        result = find_function_order(encoders, provided_encodings, required_encodings)
        assert (
            len(result) == 0
        ), "Test case 5 failed! Order of encoding functions is incorrect"

        # Test case 6: Unsatisfiable required encodings
        provided_encodings = ["aa_seq"]
        required_encodings = ["bnd_count"]  # requires "bnd_idcs"
        with pytest.raises(ValueError):
            find_function_order(encoders, provided_encodings, required_encodings)

        # Test case 7: All provided encodings are already satisfied
        provided_encodings = ["atm_count", "bnd_count"]
        required_encodings = ["atm_count", "bnd_count"]
        expected_order = []
        result = find_function_order(encoders, provided_encodings, required_encodings)
        assert (
            len(result) == 0
        ), "Test case 7 failed! Order of encoding functions is incorrect"
