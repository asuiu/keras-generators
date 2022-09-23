#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>
from abc import ABC, abstractmethod
from typing import Callable, Union, Collection

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

    @abstractmethod
    def select_features(self, features_idxs:Collection[int]) -> 'DataEncoder':
        """
        Creates a new DataEncoder with the subset of the selected features.
        It's responsibility is to adapt the encoder to the sub-set of features.
        Example: `small_ds = ds.select_features([0, 1, 2])`
        """
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

    def select_features(self, features_idxs:Collection[int]) -> 'DataEncoder':
        if isinstance(self.scaler, StandardScaler):
            scale_ = self.scaler.scale_[features_idxs]
            mean_ = self.scaler.mean_[features_idxs]
            var_ = self.scaler.var_[features_idxs]
            n_samples_seen_ = self.scaler.n_samples_seen_
            new_scaler = StandardScaler(with_mean=self.scaler.with_mean, with_std=self.scaler.with_std)
            new_scaler.scale_ = scale_
            new_scaler.mean_ = mean_
            new_scaler.var_ = var_
            new_scaler.n_samples_seen_ = n_samples_seen_
            return ScaleEncoder(new_scaler, self.column_wise)
        elif isinstance(self.scaler, MinMaxScaler):
            scale_ = self.scaler.scale_[features_idxs]
            min_ = self.scaler.min_[features_idxs]
            n_samples_seen_ = self.scaler.n_samples_seen_
            data_min_ = self.scaler.data_min_[features_idxs]
            data_max_ = self.scaler.data_max_[features_idxs]
            data_range_ = self.scaler.data_range_[features_idxs]
            new_scaler = MinMaxScaler(feature_range=self.scaler.feature_range)
            new_scaler.scale_ = scale_
            new_scaler.min_ = min_
            new_scaler.n_samples_seen_ = n_samples_seen_
            new_scaler.data_min_ = data_min_
            new_scaler.data_max_ = data_max_
            new_scaler.data_range_ = data_range_
            return ScaleEncoder(new_scaler, self.column_wise)
        raise NotImplementedError()

