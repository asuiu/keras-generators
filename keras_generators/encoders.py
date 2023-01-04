#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>
from abc import ABC, abstractmethod
from typing import Callable, Collection, List, Tuple, Union

import numpy as np
import sklearn
from sklearn.preprocessing import (
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

Scaler = Union[
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Binarizer,
    PowerTransformer,
]

_FIRST_MONDAY_TS_MS = 345600000
_DAY_MS = 24 * 3600 * 1000
_WEEK_MS = 7 * 24 * 3600 * 1000


class DataEncoder(ABC):
    def fit_encode(self, data: np.ndarray) -> np.ndarray:
        return self.encode(data)

    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def select_features(self, features_idxs: Collection[int]) -> "DataEncoder":
        """
        Creates a new DataEncoder with the subset of the selected features.
        It's responsibility is to adapt the encoder to the sub-set of features.
        Example: `small_ds = ds.select_features([0, 1, 2])`
        """
        raise NotImplementedError()

    @abstractmethod
    def clone(self) -> "DataEncoder":
        raise NotImplementedError()


class ChainedDataEncoder(DataEncoder):
    def __init__(self, encoders: List[DataEncoder]):
        self.encoders = encoders

    def fit_encode(self, data: np.ndarray) -> np.ndarray:
        encoded_na = data
        for encoder in self.encoders:
            encoded_na = encoder.fit_encode(data=encoded_na)
        return encoded_na

    def encode(self, data: np.ndarray) -> np.ndarray:
        encoded_na = data
        for encoder in self.encoders:
            encoded_na = encoder.encode(data=encoded_na)
        return encoded_na

    def decode(self, data: np.ndarray) -> np.ndarray:
        decoded_na = data
        for encoder in reversed(self.encoders):
            decoded_na = encoder.decode(data=decoded_na)
        return decoded_na

    def select_features(self, features_idxs: Collection[int]) -> "ChainedDataEncoder":
        new_encoders = [encoder.select_features(features_idxs) for encoder in self.encoders]
        return self.__class__(new_encoders)

    def clone(self) -> "ChainedDataEncoder":
        return self.__class__([encoder.clone() for encoder in self.encoders])


class CompoundDataEncoder(DataEncoder):
    def __init__(self, encoders: List[DataEncoder], instance_shapes: Collection[Tuple[int, ...]]) -> None:
        self.encoders = encoders
        self.instance_shapes = instance_shapes

    def _split_data(self, data: np.ndarray) -> List[np.ndarray]:
        nas = []
        cur_col = 0
        for shape in self.instance_shapes:
            cols = shape[-1]
            columns = data[..., cur_col : cur_col + cols]
            nas.append(columns)
            cur_col += cols
        return nas

    def fit_encode(self, data: np.ndarray) -> np.ndarray:
        encoded_nas = []
        for i, na in enumerate(self._split_data(data)):
            encoder = self.encoders[i]
            data = encoder.fit_encode(data=na)
            encoded_nas.append(data)
        encoded_na = np.concatenate(encoded_nas, axis=-1)
        return encoded_na

    def encode(self, data: np.ndarray) -> np.ndarray:
        encoded_nas = []
        for i, na in enumerate(self._split_data(data)):
            encoder = self.encoders[i]
            data = encoder.encode(data=na)
            encoded_nas.append(data)
        encoded_na = np.concatenate(encoded_nas, axis=-1)
        return encoded_na

    def decode(self, data: np.ndarray) -> np.ndarray:
        decoded_nas = []
        for i, na in enumerate(self._split_data(data)):
            encoder = self.encoders[i]
            data = encoder.decode(data=na)
            decoded_nas.append(data)
        decoded_na = np.concatenate(decoded_nas, axis=-1)
        return decoded_na

    def select_features(self, features_idxs: Collection[int]) -> "DataEncoder":
        raise NotImplementedError("select_features() doesn't make much sense for CompoundDataEncoder")

    def clone(self) -> "CompoundDataEncoder":
        new_encoders = [encoder.clone() for encoder in self.encoders]
        return self.__class__(encoders=new_encoders, instance_shapes=self.instance_shapes)


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
            if len(na.shape) == 3:
                na_to_norm = na.reshape(na.shape[0] * na.shape[1], na.shape[2])
                return transformer(na_to_norm).reshape(na.shape)
            raise ValueError(f"Unsupported shape: {na.shape}")
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

    def select_features(self, features_idxs: Collection[int]) -> "ScaleEncoder":
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
        if isinstance(self.scaler, MinMaxScaler):
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

    def clone(self) -> "ScaleEncoder":
        new_scaler = sklearn.base.clone(self.scaler)
        return self.__class__(scaler=new_scaler, column_wise=self.column_wise)


class PeriodEncoder(DataEncoder):
    """
    Encode a periodic data on a trigonometric circle using sine and cosine.
    For example, to encode time of day, use 24h period
    Note: the output of the encoder will be a doubled vector (2 * n_features)
            where the first half is sine and the second half is cosine
    """

    def __init__(self, period: float) -> None:
        if not period > 0:
            raise ValueError("Period must be positive")
        self._period = period

    def fit_encode(self, data: np.ndarray) -> np.ndarray:
        return self.encode(data)

    def encode(self, data: np.ndarray) -> np.ndarray:
        # transform the data to be in the range [0, 1]
        offset = np.modf(data / self._period)[0]
        if np.any(offset < 0):
            # to handle negative offsets, we add 1 and perform a modulo 1
            offset = np.modf(offset + 1)[0]
        time_coord_on_circle = 2 * np.pi * offset
        sin = np.sin(time_coord_on_circle)
        cos = np.cos(time_coord_on_circle)
        return np.concatenate([sin, cos], axis=-1)

    def decode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Can not decode period encoded data")

    def select_features(self, features_idxs: Collection[int]) -> "PeriodEncoder":
        return self

    def clone(self) -> "PeriodEncoder":
        return self.__class__(period=self._period)


class DayOfWeekEncoder(DataEncoder):
    """
    This transformer encodes the received timestamp in milliseconds to the corresponding day of the week.
    """

    def encode(self, data: np.ndarray) -> np.ndarray:
        dow_na = np.floor_divide(data - _FIRST_MONDAY_TS_MS, _DAY_MS).astype(np.int64) % 7
        return dow_na

    def decode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Can not decode period encoded data")

    def select_features(self, features_idxs: Collection[int]) -> "PeriodEncoder":
        # It's okay that we return self here, because the encoder is stateless
        return self

    def clone(self) -> "PeriodEncoder":
        # It's okay that we return self here, because the encoder is stateless
        return self


class WeekTSEncoder(DataEncoder):
    """
    This transformer encodes the received timestamp in milliseconds to millisecond within current week,
        i.e. number of milliseconds since Monday 00:00:00.
    """

    def encode(self, data: np.ndarray) -> np.ndarray:
        week_ts_na = (data - _FIRST_MONDAY_TS_MS) % _WEEK_MS
        return week_ts_na

    def decode(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Can not decode period encoded data")

    def select_features(self, features_idxs: Collection[int]) -> "PeriodEncoder":
        # It's okay that we return self here, because the encoder is stateless
        return self

    def clone(self) -> "PeriodEncoder":
        # It's okay that we return self here, because the encoder is stateless
        return self
