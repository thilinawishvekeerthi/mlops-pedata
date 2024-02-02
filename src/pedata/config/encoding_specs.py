""" Module for encoding specifications and adding encodings to datasets.
This module contains the encoding specifications for the encodings that are provided by the package.

The encodings are specified as a list `encodings` containing `EncodingSpec` or `SklEncodingSpec` objects.

`EncodingSpec` objects contain the following attributes:
    - `provides` (List): The names of the columns that the encoding will provide
    - `requires` (List): The names of the columns that the encoding requires
    - `func`: A function that takes a dataset and returns a dictionary of columns that the encoding provides

SklEncodingSpec objects are slightly different and contain the following attributes:
    - `provides` (str): The name of the column that the encoding will provide
    - `requires` (str): The name of the column that the encoding requires
    - `func`: A sklearn Transformer (sklearn.TransformerMixin) object that will be fit and transformed on the dataset

The `encodings` list contains all the encodings that are provided by the package and is used by 
the `add_encodings` function, which takes a dataset or a dataset dictionary and adds the specified encodings to it.
Encodings are applied in a specific order based on their dependencies.

`provided_encodings` is a list of all the encodings that are provided by the package.

>>> print(provided_encodings) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
['atm_count', 'bnd_count', 'atm_adj', 'atm_bnd_incid', 'atm_retprob100', 
'aa_unirep_1900', 'aa_unirep_final', 'aa_seq', 'aa_len', 'aa_1gram', 
'aa_ankh_base', 'aa_esm2_t6_8M', 'aa_1hot', 'dna_len', 'dna_1hot']
"""

from typing import Union
from sklearn.base import TransformerMixin
from functools import partial
import datasets as ds
from ..encoding import (
    EncodingSpec,
    SklEncodingSpec,
    NGramFeat,
    Ankh,
    ESM,
    SeqStrOneHot,
    SeqStrLen,
    unirep,
    translate_dna_to_aa_seq,
    return_prob_feat,
)
from ..encoding import transforms_graph as tg
from ..encoding.util import find_function_order
from .alphabets import (
    dna_alphabet,
    aa_alphabet,
    smiles_alphabet,
    padding_value_enc,
)


encodings: list[EncodingSpec] = [
    # EncodingSpec(["list", "of", "provided", "encodings"], ["list_of","required_encodings"], function_taking_dataset_and_returning_dict),
    EncodingSpec(["atm_count", "bnd_count"], ["bnd_idcs"], tg.bnd_count_atm_count),
    EncodingSpec(["atm_adj"], ["atm_count", "bnd_count"], tg.atm_adj),
    EncodingSpec(["atm_bnd_incid"], ["atm_count", "bnd_count"], tg.atm_bnd_incid),
    EncodingSpec(
        ["atm_retprob100"],  # shape: (atm_count, atm_count)
        ["atm_adj"],  # shape: (2, atm_count)
        partial(
            return_prob_feat,
            100,
        ),
    ),
    EncodingSpec(["aa_unirep_1900", "aa_unirep_final"], ["aa_seq"], unirep),
    EncodingSpec(["aa_seq"], ["dna_seq"], translate_dna_to_aa_seq),
    EncodingSpec(
        ["aa_ankh_avg"],
        ["aa_ankh_base"],
        lambda df: {
            "aa_ankh_avg": df.with_format("numpy")["aa_ankh_base"].mean(axis=1).tolist()
        },
    ),
    EncodingSpec(
        ["aa_esm2_avg"],
        ["aa_esm2_t6_8M"],
        lambda df: {
            "aa_esm2_avg": df.with_format("numpy")["aa_esm2_t6_8M"]
            .mean(axis=1)
            .tolist()
        },
    ),
    EncodingSpec(
        ["aa_1hot"],
        ["aa_seq"],
        SeqStrOneHot("aa_1hot", "aa_seq", padding_value_enc, aa_alphabet),
    ),
    EncodingSpec(
        ["dna_1hot"],
        ["dna_seq"],
        SeqStrOneHot("dna_1hot", "dna_seq", padding_value_enc, dna_alphabet),
    ),
    EncodingSpec(
        ["smiles_1hot"],
        ["smiles_seq"],
        SeqStrOneHot("smiles_1hot", "smiles_seq", padding_value_enc, smiles_alphabet),
    ),
    # Deprecated API below, only allows a single provided and single required encoding
    # SklEncodingSpec("provided", "required", sklearn.TransformerMixin),
    # This is not too much of an issue
    SklEncodingSpec("aa_len", "aa_seq", SeqStrLen()),  # aa_len shape: (1,)
    SklEncodingSpec("aa_1gram", "aa_seq", NGramFeat(1, aa_alphabet)),
    SklEncodingSpec("aa_ankh_base", "aa_seq", Ankh()),
    SklEncodingSpec("aa_esm2_t6_8M", "aa_seq", ESM()),
    SklEncodingSpec("dna_len", "dna_seq", SeqStrLen()),
]


provided_encodings = [p for e in encodings for p in e.provides]


def add_encodings(
    dataset_dict: Union[ds.DatasetDict, ds.Dataset],
    needed: list[str] | set[str] = [],
) -> Union[ds.DatasetDict, ds.Dataset]:
    """Add encodings to a single Dataset or Datasets in a dataset dictionary.

    This function takes a dataset dictionary or a single dataset and adds the specified encodings to it.
    Encodings are applied in a specific order based on their dependencies.

    Args:
        dataset_dict (Union[ds.DatasetDict, ds.Dataset]: Dataset or dataset dictionary to which encodings should be added
        needed (Union[list[str], set[str]], optional): List or set of encodings to be added. Defaults to None.

    Returns:
        Union[ds.DatasetDict, ds.Dataset]: Dataset or dataset dictionary with new encodings added

    Raises:
        TypeError: If the input is not a valid dataset or dictionary of datasets.
        TypeError: If the `needed` parameter is not a list or valid encodings.

    Example:
        >>> needed = ["aa_len", "aa_1gram", "aa_1hot"]
        >>> dataset = ds.Dataset.from_dict(
        ...     {
        ...         "aa_mut": ["wildtype", "L2A"],
        ...         "aa_seq": ["MLGTK", "MAGTK"],
        ...         "target foo": [1, 2],
        ...     }
        ... )
        >>> encoded = add_encodings(dataset, needed)
        >>> print(encoded) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Dataset({
            features: ['aa_mut', 'aa_seq', 'target foo', 'aa_len', 'aa_1gram', 'aa_1hot'],
            num_rows: 2
        })
    """

    # do not include by default because they are very long to be computed
    #   "aa_ankh_base", "aa_esm2_t6_8M", "aa_esm2_avg", "aa_ankh_avg", "aa_unirep_1900", "atm_retprob100"
    # FIXME: add this above in the encoding list as fast_compute = False
    none_default_encodings = [
        "aa_ankh_base",
        "aa_ankh_avg",
        "aa_esm2_avg",
        "aa_esm2_t6_8M",
        "atm_retprob100",
    ]
    # If `dataset_dict` is a dictionary, iterate over each dataset and recursively call `add_encodings`
    if isinstance(dataset_dict, ds.DatasetDict):
        for name, dataset in dataset_dict.items():
            dataset_dict[name] = add_encodings(dataset, needed)

        return dataset_dict

    dataset = dataset_dict
    if isinstance(dataset, ds.Dataset):
        require_all = True  # Will help determine how strict encodings should be

        # If no encoding was provided, apply only encodings that are satisfiable by changing "require_all" to False
        if len(needed) == 0:
            needed = list(set(provided_encodings) - set(none_default_encodings))
            require_all = False

        if not isinstance(needed, list):
            raise TypeError("Input a valid list of encodings")

        # Determine the order in which encodings should be applied based on their dependencies

        func_order = find_function_order(
            encoders=encodings,
            provided_encodings=list(dataset.features.keys()),
            required_encodings=needed,
            satisfy_all=require_all,
        )

        # Apply the encoding functions and add the resulting columns to the dataset
        for enc in func_order:
            if isinstance(enc.func, TransformerMixin):
                if len(enc.provides) != 1:
                    raise Exception(
                        "Only single column encodings supported when using the old TransformerMixin interface"
                    )

                # Apply encoding using the `map_func` if available
                if hasattr(enc.func, "map_func"):
                    dataset = dataset.map(
                        lambda x: {enc.provides[0]: enc.func.map_func(x)},
                        writer_batch_size=100,
                        batch_size=100,
                        batched=True,
                    )
                else:
                    # Fit and transform the dataset using the encoding function
                    enc.func.fit(dataset)
                    val = enc.func.transform(dataset)
                    dataset = dataset.add_column(enc.provides[0], val)

            else:
                val = enc.func(dataset)

                for prov in enc.provides:
                    if prov not in val:
                        assert f"Encoding {enc} did not provide {prov} unlike specified"

                for k in val:
                    # if the column already exists - it might happen when multiple encodings are returned together (e.g. unirep)
                    if k in list(dataset.features.keys()):
                        dataset = dataset.remove_columns(
                            k
                        )  # TODO: write a test for this
                    # adds the column
                    dataset = dataset.add_column(k, val[k])

    else:
        raise TypeError("Invalid input type. Expected Dataset or a dataset dictionary.")

    return dataset
