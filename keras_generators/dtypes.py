"""
These types are needed to differentiate between distinct types of multi-dimensional data:
Multi-sequence(time-series) data and image data are both bi-dimensional, but the scaling/normalization is performed distinctly,
thus we define these classes for every of these types.
"""

from abc import abstractmethod, ABC
from typing import Optional, Union, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .splitters import TrainValTestSpliter

Scaler = Union[StandardScaler, MinMaxScaler]


def scale_scalar_and_1D_sequences(na: np.ndarray, scaler: Optional[Scaler]) -> np.ndarray:
    if scaler is not None:
        if hasattr(scaler, "scale_"):
            scale = scaler.transform
        else:
            scale = scaler.fit_transform

        if len(na.shape) == 2:
            return scale(na)
        elif len(na.shape) == 3:
            na_to_norm = na.reshape(na.shape[0] * na.shape[1], na.shape[2])
            return scale(na_to_norm).reshape(na.shape)
        else:
            raise NotImplementedError(f"{na.shape}")
    else:
        return na


class TensorSeq(ABC):
    """
    This is base class for numeric tensor sequences, which basically represents a Data Set of instances,
    where every instance represent a tensor (either a vector of time-series - being a matrix, either a vector of scalar values)
    """

    def __init__(self, name: str, tensors: Union[Sequence[np.ndarray], np.ndarray]) -> None:
        if not isinstance(tensors, np.ndarray):
            tensors = np.array(tensors)
        self.tensors = tensors
        self.data_shape = tensors.shape[1:]
        self.name = name

    def split(self, splitter: TrainValTestSpliter) -> Tuple['TensorSeq', 'TensorSeq', 'TensorSeq']:
        train, val, test = splitter.split(self.tensors)
        return self.__class__(self.name, train), self.__class__(self.name, val), self.__class__(self.name, test)

    @abstractmethod
    def scale(self, scaler: Optional[Scaler]) -> 'ScaledTensorSeq':
        raise NotImplementedError()


class ScalarVecSeq(TensorSeq):
    """
    This is class represents sequence of numeric vectors of scalar features, commonly known as Tabular Data
    It can be served as input for Dense NN layers.
    """

    def __init__(self, name: str, tensors: Union[Sequence[np.ndarray], np.ndarray]) -> None:
        super().__init__(name, tensors)
        assert len(self.data_shape) == 1, f"Missmatched shape for {self.__class__.__name__}: {self.data_shape}"

    def scale(self, scaler: Optional[Scaler]) -> 'ScaledScalarVecSeq':
        return ScaledScalarVecSeq.from_seq(name=self.name, seq=self, scaler=scaler)


class Num1DVecSeq(TensorSeq):
    """
    This is class represents sequence of multi-feature time-series numeric data,
        where each feature is a 1D vector representing one time-series.
    It can be served as input for Conv1D or RNN sequntial types of layers.
    """

    def __init__(self, name: str, tensors: Union[Sequence[np.ndarray], np.ndarray]) -> None:
        super().__init__(name, tensors)
        assert len(self.data_shape) == 2, f"Missmatched shape for {self.__class__.__name__}: {self.data_shape}"

    def scale(self, scaler: Optional[Scaler]) -> 'Scaled1DVecSeq':
        return Scaled1DVecSeq.from_seq(name=self.name, seq=self, scaler=scaler)


class ScaledTensorSeq(ABC):
    """
    This is a base class for scaled numeric tensor sequences
    """

    def __init__(self, name: str, norm_na: np.array, scaler: Optional[Scaler]) -> None:
        self.norm_na = norm_na
        self.name = name
        self.scaler = scaler

    @classmethod
    def from_seq(cls, name: str, seq: TensorSeq, scaler: Optional[Scaler]) -> 'ScaledTensorSeq':
        if scaler is not None:
            norm_na = scale_scalar_and_1D_sequences(seq.tensors, scaler)
        else:
            norm_na = seq.tensors

        return cls(name=name, norm_na=norm_na, scaler=scaler)

    @abstractmethod
    def get_original_tensors(self) -> np.array:
        raise NotImplementedError()

    def __add__(self, other: 'ScaledTensorSeq') -> 'ScaledTensorSeq':
        assert isinstance(other, self.__class__), f'{type(other)} is not a {self.__class__.__name__}'
        assert self.name == other.name
        if self.scaler is None:
            assert other.scaler is None
        else:
            assert (self.scaler.mean_, self.scaler.scale_) == (other.scaler.mean_, other.scaler.scale_)
        norm_na = np.concatenate((self.norm_na, other.norm_na), axis=0)
        return self.__class__(name=self.name, norm_na=norm_na, scaler=self.scaler)


class ScaledScalarVecSeq(ScaledTensorSeq):
    """
    This is class represents sequence of numeric vectors of scalar features, , commonly known as Tabular Data.
    It can be served as input for Dense NN layers.

    The data is already scaled and normalized according to the Scaler received by constructor.
    """

    def get_original_tensors(self) -> np.ndarray:
        if self.scaler is None:
            return self.norm_na
        return self.scaler.inverse_transform(self.norm_na, copy=True)


class Scaled1DVecSeq(ScaledTensorSeq):
    """
    This is class represents sequence of multi-feature time-series numeric data,
        where each feature is a 1D vector representing one time-series.
    It can be served as input for Conv1D or RNN sequntial types of layers.

    The data is already scaled and normalized according to the Scaler received by constructor.
    """

    def get_original_tensors(self) -> np.array:
        na = self.norm_na
        if self.scaler is None:
            return na
        na_to_invert = na.reshape(na.shape[0] * na.shape[1], na.shape[2])
        return self.scaler.inverse_transform(na_to_invert).reshape(na.shape)
