from dataclasses import dataclass
from typing import Union, Callable

from sklearn.base import TransformerMixin
from .transform import FixedSingleColumnTransform


@dataclass
class EncodingSpec(object):
    """Specification for column encoding."""

    provides: Union[list[str], str]  # list of encodings provided by the transformer
    requires: Union[list[str], str]  # list of encodings required by the transformer
    func: Union[
        TransformerMixin, Callable  # FIXME remove TransformerMixin
    ]  # transformer/function to be applied to source file

    def __post_init__(self):
        """Post init function – ensures that provides and requires are lists"""
        if isinstance(self.provides, str):
            self.provides = [self.provides]
        if isinstance(self.requires, str):
            self.requires = [self.requires]


@dataclass
class SklEncodingSpec(EncodingSpec):
    """A specification for column encodings using SKLearn transformers. Allows a single requirement and a single provided encoding."""

    def __post_init__(self):
        """Post init function – adds correct inner transformer automatically"""
        super().__post_init__()
        self.func = FixedSingleColumnTransform(self.func, self.requires[0])
        if hasattr(self.func, "map_func"):
            self.map_func = self.func.map_func
