#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>
from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Binarizer, PowerTransformer

Scaler = Union[StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Binarizer, PowerTransformer]


class DataEncoder(ABC):
    @abstractmethod
    def fit_encode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class ScaleEncoder(DataEncoder):
    """
    Note: when working with 1D vector sequences (multivariate time-series),
    the shape of the input data should be (n_samples, n_timesteps, n_features)
    LSTM uses  [batch, timesteps, feature].
    """

    def __init__(self, scaler: Scaler, column_wise: bool = True) -> None:
        self.scaler = scaler
        self.column_wise = column_wise

    def _transform(self, data: np.ndarray, transformer: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        na = data
        if self.column_wise:
            if len(na.shape) == 2:
                return transformer(data)
            elif len(na.shape) == 3:
                na_to_norm = na.reshape(na.shape[0] * na.shape[1], na.shape[2])
                return transformer(na_to_norm).reshape(na.shape)
            else:
                raise ValueError(f'Unsupported shape: {na.shape}')
        na_to_norm = na.reshape(na.shape[0], 1)
        return transformer(na_to_norm).reshape(na.shape)

    def fit_encode(self, data: np.ndarray) -> np.ndarray:
        transformer = self.scaler.fit_transform
        return self._transform(data, transformer)

    def encode(self, data: np.ndarray) -> np.ndarray:
        transformer = self.scaler.transform
        return self._transform(data, transformer)

    def decode(self, data: np.ndarray) -> np.ndarray:
        transformer = self.scaler.inverse_transform
        return self._transform(data, transformer)
