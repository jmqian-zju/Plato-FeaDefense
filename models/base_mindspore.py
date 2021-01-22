"""Base classes for MindSpore models."""

from abc import ABC, abstractmethod, abstractstaticmethod
import mindspore.nn as nn


class Model(ABC, nn.Cell):
    """The base class for by all the models."""
    @abstractmethod
    def constrct(self, x):
        """The forward pass."""

    @abstractstaticmethod
    def is_valid_model_name(model_name: str) -> bool:
        """Is the model name string a valid name for models in this class?"""

    @abstractstaticmethod
    def get_model_from_name(model_name: str) -> 'Model':
        """Returns an instance of this class as described by the model_name string."""