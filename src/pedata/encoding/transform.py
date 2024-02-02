from typing import Iterable, Sequence, Tuple, Union

import datasets as ds
import numpy as onp
import pandas as pd
import sklearn
import sklearn.feature_extraction
import sklearn.pipeline
import sklearn.preprocessing
from jax_unirep import get_reps
from sklearn.base import BaseEstimator, TransformerMixin

from ..config.alphabets import padding_value_enc
from Bio.Seq import Seq


def array_to_list_of_arrays(x: onp.ndarray):
    """Convert a 2d array into a list of arrays."""
    return [r for r in x]


def seq_strings_to_array(
    seq_strings: Union[onp.ndarray, list[str]], pad=False, pad_char="?"
) -> onp.array:
    """Convert strings into an array.
    If needed a padding can be added.

    Args:
        seq_strings (Union[onp.ndarray, list[str]]): List or array of sequences
        pad (bool): if padding should be done (default is False)
        pad_char (str):  padding character  (default is '?')

    Returns:
        onp.array : array of the sequences
    """
    assert isinstance(seq_strings, list) or len(seq_strings.shape) == 1
    tmp = []
    all_length = {len(seq) for seq in seq_strings}
    max_length = None
    if pad:
        max_length = max(all_length)
    else:
        assert (
            len(all_length) == 1
        ), "pad == False, but found sequences strings of differing length"

    for seq in seq_strings:
        seq_padding = [pad_char] * (max_length - len(seq))
        tmp.append(onp.array(list(seq) + seq_padding))

    return onp.array(tmp)


def unirep(df: Union[ds.Dataset, pd.DataFrame]) -> dict[str, onp.ndarray]:
    """Add unirep encoding to a dataframe.

    Args:
        df (Union[ds.Dataset, pd.DataFrame]) : Data Frame or data set to which encodings should be added

    Returns:
        rval (ds.Dataset): DataFrame with new encodings added
    """
    h_avg, h_final, c_final = get_reps(df["aa_seq"])
    return {"aa_unirep_1900": h_avg.tolist(), "aa_unirep_final": h_final.tolist()}


def translate_dna_to_aa_seq(dataset: ds.Dataset) -> ds.Dataset:
    """Translate DNA sequences to amino acid sequences.

    This function takes a Hugging Face dataset containing DNA sequences and translates them into
    amino acid sequences using the Biopython library. It creates a new column 'aa_seq' in the dataset
    which contains the translated amino acid sequences.

    Applying translate_dna_to_aa_seq(dataset) will modify the dataset by adding a new column 'aa_seq'
    which contains the translated amino acid sequences. Here's an example of the input dataset and
    the resulting dataset after applying the function:

    Input dataset:
    +----------------+
    |   dna_seq      |
    +----------------+
    |  GACCTA        |
    |  GAGCCA        |
    |  GTCGTC        |
    +----------------+

    Resulting dataset:
    +----------------+-----------------+
    |   dna_seq      |     aa_seq      |
    +----------------+-----------------+
    |  GACCTA        |   DL            |
    |  GAGCCA        |   EP            |
    |  GTCGTC        |   VV            |
    +----------------+-----------------+


    Args:
        dataset (datasets.Dataset): Hugging Face dataset with DNA sequences.

    Returns:
        datasets.Dataset: Dataset with amino acid sequences added.

    Example:
        >>> dataset = ds.Dataset.from_dict({"dna_seq": ["GATCTG"]})
        >>> translate_dna_to_aa_seq(dataset)
        {'aa_seq': ['DL']}

    """

    # Check if input is a dataset
    if isinstance(dataset, ds.Dataset):
        # Check if required column exists
        if "dna_seq" not in list(dataset.features.keys()):
            raise TypeError(
                f"Invalid input! Expected a valid huggingface dataset but got a {type(dataset)}"
            )

        # Use dataset.map() to apply the translation to each example in the dataset
        encoded_dataset = dataset.map(
            lambda example: {
                "aa_seq": str(Seq(example["dna_seq"]).translate()).strip("*")
            }
        )

    else:
        raise TypeError(
            f"Invalid input! Expected a valid huggingface dataset but got a {type(dataset)}"
        )

    # Return encodings
    return {"aa_seq": encoded_dataset["aa_seq"]}


class FixedSingleColumnTransform(TransformerMixin):
    """SKLearn Transformer that transforms a single column of a dataframe"""

    def __init__(self, inner_transformer: TransformerMixin, requires: str) -> None:
        """Constructor for FixedSingleColumnTransform.

        Args:
            inner_transformer (TransformerMixin): The transformer to apply to the column.
            column_name (str): The name of the column to apply the transformer to.
        """
        self.column_name = requires
        self.inner_transformer = inner_transformer
        if hasattr(self.inner_transformer, "map_func"):
            self.map_func = self.inner_transformer.map_func

    def drop_non_column(
        self, X: Union[pd.DataFrame, ds.Dataset]
    ) -> Union[pd.DataFrame, ds.Dataset]:
        """Drop all columns except the one to transform

        Args:
            X (Union[pd.DataFrame, ds.Dataset]): The dataframe or dataset to drop columns from.

        Returns:
            Union[pd.DataFrame, ds.Dataset]: The dataframe or dataset with only the column to transform.
        """
        if isinstance(X, ds.Dataset):
            X = X.with_format("numpy")
        if isinstance(X, pd.DataFrame) or isinstance(X, ds.Dataset):
            return X[self.column_name]
        else:
            return X

    def fit(
        self, X: Union[pd.DataFrame, ds.Dataset], **fit_params
    ) -> "FixedSingleColumnTransform":
        """Fit the transformer.

        Args:
            X (Union[pd.DataFrame, ds.Dataset]): The dataframe or dataset to fit the transformer on.

        Returns:
            FixedSingleColumnTransform: The fitted transformer.
        """
        self.inner_transformer.fit(self.drop_non_column(X), **fit_params)
        return self.__class__

    def transform(
        self, X: Union[pd.DataFrame, ds.Dataset]
    ) -> Union[pd.DataFrame, ds.Dataset]:
        """Transform the dataframe or dataset.

        Args:
            X (Union[pd.DataFrame, ds.Dataset]): The dataframe or dataset to transform.

        Returns:
            Union[pd.DataFrame, ds.Dataset]: The transformed dataframe or dataset.
        """
        return self.inner_transformer.transform(self.drop_non_column(X))


class SeqStrLen(sklearn.preprocessing.FunctionTransformer):
    """Transformer computing the length of a sequence string"""

    def __init__(
        self,
    ):
        """Constructor"""
        super().__init__(self.__seqlen)

    def __seqlen(self, seq_strings: Sequence[str]) -> onp.ndarray:
        """Compute the length of each string in a sequence of strings.

        Args:
            seq_strings (Sequence[str]): The sequence of strings.

        Returns:
            onp.ndarray: The length of each string in the sequence as a numpy array.
        """
        all_length = [len(seq) for seq in seq_strings]
        return onp.array(all_length, dtype=onp.int32)  # .reshape(-1, 1)


class Unirep1900(sklearn.preprocessing.FunctionTransformer):
    """Transformer computing the UniRep 1900 representation of sequence strings"""

    def __init__(self):
        """Constructor"""
        # el.strip().strip("*").replace("*", "M")
        super().__init__(lambda x: [get_reps(el.strip())[0].squeeze() for el in x])

    def map_func(self, x: Union[pd.DataFrame, ds.Dataset]) -> onp.ndarray:
        """Compute the UniRep 1900 representation of a sequence string.

        Args:
            x (Union[pd.DataFrame, ds.Dataset]): The dataset containing AA sequence strings.

        Returns:
            onp.ndarray: The UniRep 1900 representation of the sequence strings.
        """
        return get_reps(x["aa_seq"])[0]


class Unirep1900Final(sklearn.preprocessing.FunctionTransformer):
    """Transformer computing the UniRep 1900 representation of sequence strings"""

    def __init__(self):
        """Constructor"""
        # el.strip().strip("*").replace("*", "M")
        super().__init__(lambda x: [get_reps(el.strip())[0].squeeze() for el in x])

    def map_func(self, x: Union[pd.DataFrame, ds.Dataset]) -> onp.ndarray:
        """Compute the UniRep 1900 representation of a sequence string.

        Args:
            x (Union[pd.DataFrame, ds.Dataset]): The dataset containing AA sequence strings.

        Returns:
            onp.ndarray: The UniRep 1900 representation of the sequence strings.
        """
        return get_reps(x["aa_seq"])[1]


class GuaranteeShapeTransformer(BaseEstimator, TransformerMixin):
    """Transformer that guarantees a certain shape of the output."""

    def __init__(self, wrapped: BaseEstimator) -> None:
        """Constructor for GuaranteeShapeTransformer transformer

        Args:
            wrapped (BaseEstimator): The estimator to be wrapped
        """
        super().__init__()
        self.wrapped = wrapped

    def fit(self, X, y=None):
        """Fit the wrapped estimator

        Args:
            X (pd.DataFrame): Data to be fitted
            y (pd.Series): Variable of interest. Defaults to None.
        """
        self.wrapped.fit(X.reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        """Transform the data using the wrapped estimator, then reshape the output.

        Args:
            X (pd.DataFrame): Data to be fitted
            y (pd.Series): Variable of interest. Defaults to None.
        """
        return self.wrapped.transform(X.reshape(-1, 1)).reshape(X.shape[0], -1)


class SeqStrOneHot:
    """Class for one hot encoding of sequence data"""

    def __init__(
        self, provided: str, required: str, padding_char: str, alphabet: list[str]
    ) -> None:
        """Constructor for SeqStrOneHot transformer
        Args:
            alphabet (List[str]): List of the used alphabet
        """
        super().__init__()

        assert alphabet is not None
        self.alphabet = alphabet
        self.padding_char = padding_char
        self.provided = provided
        self.required = required

    def __call__(self, X: ds.Dataset) -> dict[str, list[onp.ndarray]]:
        """Function for transforming a new dataframe/dataset with the SeqStrOneHot encoder

        Args:
            X: Data to be fitted

        Returns:
            One hot encoded dataframe of X
        """
        X = X.with_format(None)[self.required]

        one_hot = sklearn.preprocessing.OneHotEncoder(
            categories=[[self.padding_char] + self.alphabet], drop="first"
        )
        one_hot.fit(seq_strings_to_array(X, True, padding_value_enc).reshape(-1, 1))
        seq_array = seq_strings_to_array(X, True, padding_value_enc)
        rval = (
            one_hot.transform(seq_array.reshape(-1, 1))
            .toarray()
            .reshape(list(seq_array.shape) + [-1])
            .astype(onp.uint8)
            .tolist()
        )
        return {self.provided: rval}


class NGramFeat(BaseEstimator, TransformerMixin):
    """Transformer for calculating ngrams of the sequence strings in a dataset"""

    def __init__(
        self,
        ngram_range: Union[int, Tuple[int, int]] = 1,
        vocabulary: Union[dict[str, int], Iterable] = None,
    ):
        """Constructor for NGramFeat transformer

        Args:
            ngram_range (Union[int, Tuple[int, int]]): The range of ngrams to be calculated.
                If a tuple, this is expected to define minimum and maximum n (min_n, max_n),
                if an int, min_n and max_n are set to the same value.
                Defaults to 1.
            vocabulary (Union[dict[str, int], Iterable]): The vocabulary to be used.
                If None, the vocabulary is built from the data.
                Defaults to None.
        """
        if not isinstance(ngram_range, tuple):
            ngram_range = (ngram_range, ngram_range)
        self.cv = sklearn.feature_extraction.text.CountVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            lowercase=False,
            vocabulary=vocabulary,
        )

    def fit(self, X, y=None) -> "NGramFeat":
        """Fit the NGramFeat transformer

        Args:
            X: Data to be fitted
            y (optional): Variable of interest. Defaults to None.

        Returns:
            NGramFeat: The fitted NGramFeat transformer
        """
        return self.cv.fit(X, y)

    def transform(self, X, y=None) -> list[onp.ndarray]:
        """Transform the data using the NGramFeat transformer

        Args:
            X: Data to be fitted
            y (optional): Variable of interest. Defaults to None.

        Returns:
            list[onp.ndarray]: The transformed data
        """
        return array_to_list_of_arrays(self.cv.transform(X).toarray())
