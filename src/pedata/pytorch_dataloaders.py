import torch
import numpy
from .encoding.embeddings import AnkhBatched, ESM
import random
import math
from .config.encoding_specs import add_encodings
from typing import Union


class Transformer:
    """
    A dummy class to be used as a placeholder for the transformer in Encoder class.
    """

    def __init__(self) -> None:
        """
        Initializes the transformer.
        """
        pass

    def transform(self, x: Union[torch.Tensor, numpy.ndarray, list]) -> torch.Tensor:
        """
        Returns the input as it is. Removes unnecessary dimensions.

        Args:
            x: Input data.

        Returns:
            Pytorch tensor of the input data.
        """
        if isinstance(x, Union[torch.Tensor, numpy.ndarray]):
            if len(x.shape) == 1:
                return torch.tensor(x).unsqueeze(-1)
            return x
        else:
            return torch.tensor(x).unsqueeze(-1)


class Encoder:
    """
    A class to encode the input data.

    Example:
        >>> encoder = Encoder(embedding_name='aa_seq')
        >>> x = {'aa_seq': 'hello'}
        >>> encoder.transform(x)
        'hello'
    """

    def __init__(
        self,
        in_embedding_name: str = None,
        out_embedding_name: str = None,
        transformer=None,
    ) -> None:
        """
        Initializes the encoder.

        Args:
            in_embedding_name: The name of the input embedding.
            out_embedding_name: The name of the output embedding.
            transformer: The transformer to be used.
        """
        if in_embedding_name is None:
            raise Exception("in_embedding_name cannot be None")
        else:
            self.in_embedding_name = in_embedding_name

        if out_embedding_name is None:
            self.out_embedding_name = in_embedding_name
        else:
            self.out_embedding_name = out_embedding_name

        if transformer is None:
            self.transformer = Transformer()
        else:
            self.transformer = transformer

    def transform(self, x: dict) -> torch.Tensor:
        """
        Returns the transformed input data.

        Args:
            x: Input data.

        Returns:
            Transformed input data as a Pytorch tensor.
        """
        return self.transformer.transform(x[self.in_embedding_name])


class Dataloader(torch.utils.data.Dataset):
    """
    A class to create a dataloader for the given dataset.
    """

    def __init__(
        self,
        dataset,
        embedding_names: list[str],
        targets: list[str] = None,
        device: torch.device = None,
        batch_size: int = 32,
        shuffle: bool = True,
        batch_norm: bool = False,
        seed: int = 42,
    ) -> None:
        """
        Initializes the dataloader.

        Args:
            dataset: The dataset to be used.
            embedding_names: The names of the encodings to be used.
            targets: The targets to be used.
            device: The device to be used.
            batch_size: The batch size to be used.
            shuffle: Whether to shuffle the data.
            batch_norm: Whether you use batch norm in your model
            seed: The seed to be used for shuffling.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.embedding_names = embedding_names
        self.seed = seed
        self.shuffle = shuffle
        self.encoders = self._get_encoders()
        self.targets = targets
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if (
            len(self.dataset) - (len(self.dataset) // self.batch_size) * self.batch_size
            == 1
            and self.targets is not None
            and batch_norm
        ):
            self.shuffle_map = list(range(len(dataset) - 1))
        else:
            self.shuffle_map = list(range(len(dataset)))

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return math.ceil(len(self.dataset) / self.batch_size)

    def _get_encoder(self, embedding_name: str) -> Encoder:
        """
        Returns the encoder for the given embedding name.

        Args:
            embedding_name: The name of the embedding.

        Returns:
            The Encoder object for the given embedding name.
        """
        if "esm" in embedding_name:
            encoder = Encoder(
                in_embedding_name="aa_seq",
                out_embedding_name="esm",
                transformer=ESM(),
            )
        elif "ankh" in embedding_name:
            encoder = Encoder(
                in_embedding_name="aa_seq",
                out_embedding_name="ankh",
                transformer=AnkhBatched(),
            )
        else:
            if embedding_name not in self.dataset.features:
                self.dataset = add_encodings(self.dataset, [embedding_name])

            encoder = Encoder(in_embedding_name=embedding_name)

        return encoder

    def _get_encoders(self) -> list[Encoder]:
        """
        Returns the encoders for the given embedding names.

        Returns:
            A list of Encoder objects.
        """
        encoders = []
        for embedding_name in self.embedding_names:
            encoders.append(self._get_encoder(embedding_name))
        return encoders

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Returns the batch data for the given index.

        Args:
            index: The index of the batch.

        Returns:
            The batch data as a dictionary of Pytorch tensors. Keys are the embedding names.
        """
        if index == 0 and self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.shuffle_map)
        elif index >= len(self):
            raise StopIteration

        index = self.shuffle_map[
            index
            * self.batch_size : min(
                (index + 1) * self.batch_size, len(self.shuffle_map)
            )
        ]

        X = {}
        y = {}

        for encoder in self.encoders:
            X[encoder.out_embedding_name] = torch.tensor(
                encoder.transform(self.dataset[index]),
                dtype=torch.float32,
                device=self.device,
            )

        if self.targets is None:
            return X
        else:
            for target in self.targets:
                y[target] = torch.tensor(
                    self.dataset[index][target], dtype=torch.float32, device=self.device
                ).unsqueeze(-1)
            return X, y
