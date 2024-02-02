from sklearn.base import BaseEstimator, TransformerMixin
import torch
import ankh
import esm
from typing import Iterable


class ESM(BaseEstimator, TransformerMixin):
    """
    ESM (Evolutionary Scale Modeling) transformer for sequence transformation using a pre-trained model.

    This class provides a transformer interface to apply the ESM model for sequence transformation tasks.

    Example:
        >>> input_sequences = ["ATGC", "GCTA"]
        >>> esm = ESM()
        >>> transformed_sequences = esm.transform(input_sequences)
        >>> print(transformed_sequences[-1][-1][-1])
        -0.03196815401315689
    """

    def __init__(self) -> None:
        """
        Initializes the ESM transformer by loading the pre-trained model and its tokenizer.
        """

        self.model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.tokenizer = alphabet.get_batch_converter()

    def fit(self, X):
        """
        Returns the pre-trained model without modifications.
        It helps to adhere to the standard API of an SK learn transformer class

        Args:
            X: Input data, not used in this method.

        Returns:
            The pre-trained model.
        """

        return self.model

    def transform(self, X: Iterable[str]) -> list:
        """
        Transforms the input sequences into their ESM representations.

        Args:
            X: Input an iterable object of strings.

        Returns:
            Transformed representations of the input sequences as a nested list.
        """

        # Check if X is not an iterable object of strings
        if (
            isinstance(X, str)  # a string is iterable, so this is needed
            or not isinstance(X, Iterable)
            or len(X) == 0
            or not all(isinstance(seq, str) for seq in X)
        ):
            raise ValueError("Input X must be a non-empty list of strings.")

        # Convert the input sequences into a format compatible with the ESM model
        sequence = [("", seq) for seq in X]

        # Convert the sequence pairs into batch tokens using the tokenizer
        _, _, batch_tokens = self.tokenizer(sequence)

        with torch.no_grad():
            # Execute the pre-trained model with the batch tokens and retrieve representations from the 6th layer
            model_outputs = self.model(
                batch_tokens, repr_layers=[6], return_contacts=True
            )

        # Return the transformed representations
        return model_outputs["representations"][6].numpy().tolist()


class Ankh(BaseEstimator, TransformerMixin):
    """
    Ankh transformer for sequence transformation using a pre-trained model.

    The Ankh class provides a transformer interface to apply the Ankh model for sequence transformation tasks.
    It utilizes a pre-trained model and tokenizer to convert input sequences into their Ankh representations.

    Usage:
        ankh = Ankh()
        transformed_sequences = ankh.transform(input_sequences)

    Example:
        >>> input_sequences = ["MAPCT", "KPGAT"]
        >>> ankh = Ankh()
        >>> transformed_sequences = ankh.transform(input_sequences)
        >>> print(transformed_sequences[-1][-1][-1])
        -0.010837498120963573
    """

    def __init__(self) -> None:
        """
        Initializes the Ankh transformer by loading the pre-trained model and its tokenizer.
        """

        self.model, self.tokenizer = ankh.load_base_model()

    def fit(self, X):
        """
        Returns the pre-trained model without modifications.
        This method is included to adhere to the standard API of an Scikit-learn transformer class.

        Args:
            X: Input data, not used in this method.

        Returns:
            The pre-trained model.
        """
        return self.model

    def transform(self, X: Iterable[str]) -> list:
        """
        Transforms the input sequences into their Ankh representations.

        Args:
            X: Input an iterable object of strings.

        Returns:
            Transformed representations of the input sequences as a nested list.
        """

        # Check if X is not an iterable object of strings
        if (
            isinstance(X, str)  # a string is iterable, so this is needed
            or not isinstance(X, Iterable)
            or len(X) == 0
            or not all(isinstance(seq, str) for seq in X)
        ):
            raise ValueError("Input X must be a non-empty list of strings.")

        # Convert the input sequences into a format compatible with the Ankh model
        sequence = [list(seq) for seq in X]

        # Convert the sequence into batch tokens using the tokenizer
        outputs = self.tokenizer.batch_encode_plus(
            sequence,
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )

        # Generate embeddings using the pre-trained model and input tokens
        with torch.no_grad():
            embeddings = self.model(
                input_ids=outputs["input_ids"], attention_mask=outputs["attention_mask"]
            )

        # Return the transformed representations
        return embeddings["last_hidden_state"].numpy().tolist()


class AnkhBatched(BaseEstimator, TransformerMixin):
    """
    Ankh batch transformer for sequence transformation using a pre-trained model.

    The Ankh class provides a transformer interface to apply the Ankh model for sequence transformation tasks.
    It utilizes a pre-trained model and tokenizer to convert input sequences into their Ankh representations.

    The difference between Ankh and AnkhBatched is that AnkhBatched sends a batch of sequences to Ankh model at once.
    It is faster than Ankh but requires more memory.
    Also the outputs are padded to the same length. This avoids the need for manually padding in the downstream tasks.

    Usage:
        ankh = Ankh()
        transformed_sequences = ankh.transform(input_sequences)

    Example:
        >>> input_sequences = ["MAPCT", "KPGAT"]
        >>> ankh = Ankh()
        >>> transformed_sequences = ankh.transform(input_sequences)
        >>> print(transformed_sequences[-1][-1][-1])
        -0.010837498120963573
    """

    def __init__(self) -> None:
        """
        Initializes the Ankh transformer by loading the pre-trained model and its tokenizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = ankh.load_base_model()
        self.model = self.model.to(self.device)

    def fit(self, X):
        """
        Returns the pre-trained model without modifications.
        This method is included to adhere to the standard API of an Scikit-learn transformer class.

        Args:
            X: Input data, not used in this method.

        Returns:
            The pre-trained model.
        """
        return self.model

    def transform(self, X: Iterable[str]) -> list:
        """
        Transforms the input sequences into their Ankh representations.

        Args:
            X: Input an iterable object of strings.

        Returns:
            Transformed representations of the input sequences as a nested list.
        """
        # Check if X is not an iterable object of strings
        if (
            not isinstance(X, Iterable)
            or len(X) == 0
            or not all(isinstance(seq, str) for seq in X)
        ):
            raise ValueError("Input X must be a non-empty list of strings.")

        sequence = [list(seq) for seq in X]
        outputs = self.tokenizer.batch_encode_plus(
            sequence,
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeddings = self.model(
                input_ids=outputs["input_ids"].to(self.device),
                attention_mask=outputs["attention_mask"].to(self.device),
            )
        output = embeddings["last_hidden_state"].cpu().numpy().tolist()
        return output

    def map_func(self, X: Iterable[str]) -> list:
        """
        This supports the batch encoding option of config.encoding_specs.add_encodings method.
        It is a wrapper for the transform method.

        Args:
            X: Input an iterable object of strings.

        Returns:
            Transformed representations of the input sequences as a nested list.
        """
        return self.transform(X["aa_seq"])
