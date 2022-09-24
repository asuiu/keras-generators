import json
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Optional, Tuple, List, Union, Sequence, Collection, ForwardRef

import numpy as np
from keras.utils import data_utils
from numpy.random import MT19937
from pydantic import PositiveInt, conint, validate_arguments
from pydantic.dataclasses import dataclass

from .common import NumpyArrayEncoder, ImmutableConfig, ArbitraryTypes
from .encoders import DataEncoder
from .splitters import TrainValTestSpliter, OrderedSplitter

"""
Reason why not use tf.Dataset:
 - when scaling data, scaling factor aren't saved, so it's impossible to scale during inference in production
"""


class DataSource(ABC):

    def __init__(self, name: str, encoders: Optional[List[DataEncoder]] = None):
        self.name = name
        self.encoders = encoders or []

    @abstractmethod
    def split(self, splitter: TrainValTestSpliter) -> Tuple['DataSource', 'DataSource', 'DataSource']:
        raise NotImplementedError()

    def __add__(self, other: 'DataSource') -> 'DataSource':
        assert isinstance(other, self.__class__), f'{type(other)} is not a {self.__class__.__name__}'
        raise NotImplementedError()

    @abstractmethod
    def fit_encode(self, encoders: List[DataEncoder]) -> 'DataSource':
        raise NotImplementedError()

    def get_encoders(self) -> List[DataEncoder]:
        return self.encoders

    @abstractmethod
    def encode(self, encoders: List[DataEncoder]) -> 'DataSource':
        raise NotImplementedError()

    @abstractmethod
    def decode(self) -> 'DataSource':
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, item: Union[int, slice, Sequence[int], np.ndarray]) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_by_idx_set(self, index_set: Collection[int]) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def as_numpy(self) -> np.ndarray:
        return np.array(self)

    @abstractmethod
    def select_features(self, features: Collection[int]) -> 'DataSource':
        """
        Creates a new DataSource with the subset of the selected features.
        It's responsibility is to adapt the encoders as well to work on new DataSource properly.
        Example: `small_ds = ds.select_features([0, 1, 2])`
        """
        raise NotImplementedError()


# ForwardRef is required by PyDantic validators for self return type for Python <3.11. Python 3.11 solves this with PEP 673
TensorDataSource = ForwardRef('TensorDataSource')


class TensorDataSource(DataSource):
    """
    Note that vectors of 1D features (i.e. multi-variate timeseries) are represented as (n_batch, n_timesteps, n_features)
    """

    def __init__(self,
                 name: str,
                 tensors: Union[np.ndarray, Sequence[np.ndarray]],
                 encoders: Optional[List[DataEncoder]] = None
                 ):
        super().__init__(name, encoders)
        if not isinstance(tensors, np.ndarray):
            tensors = np.array(tensors)
        self.tensors = tensors

    def as_numpy(self) -> np.ndarray:
        return self.tensors

    @validate_arguments(config=ArbitraryTypes)
    def split(self, splitter: TrainValTestSpliter) -> Tuple['TensorDataSource', 'TensorDataSource', 'TensorDataSource']:
        train, val, test = splitter.split(self.tensors)
        return self.__class__(self.name, train), self.__class__(self.name, val), self.__class__(self.name, test)

    @validate_arguments(config=ArbitraryTypes)
    def _clone_with_tensors_encoders(self,
                                     tensors: np.ndarray,
                                     encoders: Optional[List[DataEncoder]] = None,
                                     name: Optional[str] = None
                                     ) -> 'TensorDataSource':
        if encoders is None:
            encoders = self.encoders
        name = name or self.name
        return self.__class__(name=name, tensors=tensors, encoders=encoders)

    @validate_arguments(config=ArbitraryTypes)
    def fit_encode(self, encoders: List[DataEncoder]) -> 'TensorDataSource':
        if not encoders:
            return self
        encoded_na = self.tensors
        for encoder in encoders:
            encoded_na = encoder.fit_encode(data=encoded_na)
        if self.encoders is not None:
            encoders = self.encoders + encoders
        return self._clone_with_tensors_encoders(encoded_na, encoders)

    @validate_arguments(config=ArbitraryTypes)
    def encode(self, encoders: List[DataEncoder]) -> 'TensorDataSource':
        if not encoders:
            return self
        encoded_na = self.tensors
        for encoder in encoders:
            encoded_na = encoder.encode(data=encoded_na)
        if self.encoders is not None:
            encoders = self.encoders + encoders
        return self._clone_with_tensors_encoders(encoded_na, encoders)

    def decode(self) -> 'TensorDataSource':
        if self.encoders is None:
            return self
        decoded_na = self.tensors
        for encoder in reversed(self.encoders):
            decoded_na = encoder.decode(data=decoded_na)
        return self._clone_with_tensors_encoders(decoded_na, encoders=[])

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, item: Union[int, slice, Collection[int]]) -> np.ndarray:
        if isinstance(item, (slice, int, Collection)):
            return self.tensors[item]
        raise ValueError(f'Unsupported type: {type(item)}')

    @validate_arguments(config=ArbitraryTypes)
    def get_by_idx_set(self, index_set: Collection[int]) -> np.ndarray:
        return self.tensors[index_set]

    @validate_arguments(config=ArbitraryTypes)
    def select_features(self, features: Collection[int], name: Optional[str] = None) -> 'TensorDataSource':
        tensors = self.tensors[:, features]
        new_encoders = [encoder.select_features(features) for encoder in self.encoders]
        return self._clone_with_tensors_encoders(tensors, new_encoders, name)


@dataclass(config=ImmutableConfig)
class TimeseriesTargetsParams:
    delay: conint(ge=0) = 0
    pred_len: PositiveInt = 1
    stride: PositiveInt = 1
    target_idx: int = 0
    reverse: bool = False

    def get_raw_target_len(self) -> int:
        return self.delay + 1 + (self.pred_len - 1) * self.stride

# ForwardRef is required by PyDantic validators for self return type for Python <3.11. Python 3.11 solves this with PEP 673
TimeseriesDataSource = ForwardRef('TimeseriesDataSource')


class TimeseriesDataSource(TensorDataSource):
    """
    To make the retrieval of individual items cache friendly, we store data in (n_steps, n_feat) shape.
    LSTM uses  [batch, timesteps, feature].
    """

    @validate_arguments(config=ArbitraryTypes)
    def __init__(self,
                 name: str,
                 tensors: Union[np.ndarray, Sequence[np.ndarray]],
                 length: int,
                 sampling_rate: int = 1,
                 stride: int = 1,
                 reverse: bool = False,
                 encoders: Optional[List[DataEncoder]] = None,
                 target_params: Optional[TimeseriesTargetsParams] = None
                 ):
        super().__init__(name, tensors, encoders)
        assert len(tensors.shape) == 2
        assert length > 0
        self.length = length
        assert sampling_rate > 0
        self.sampling_rate = sampling_rate
        assert stride > 0
        self.stride = stride
        self.reverse = reverse
        self._target_params = target_params
        if target_params is not None:
            target_raw_len = target_params.get_raw_target_len()
        else:
            target_raw_len = 0
        self.w_sz = 1 + (self.length - 1) * self.sampling_rate
        useful_n = len(self.tensors) - target_raw_len
        instances = useful_n - self.w_sz + 1
        assert instances > 0, f'Not enough data for window size {self.length} and sampling rate {sampling_rate}'
        strided_instances = (instances - 1) // self.stride + min(instances, 1)
        self.size = strided_instances
        assert self.size > 0

    @validate_arguments(config=ArbitraryTypes)
    def _clone_with_tensors_encoders(self,
                                     tensors: np.ndarray,
                                     encoders: Optional[List[DataEncoder]] = None,
                                     name: Optional[str] = None
                                     ) -> 'TimeseriesDataSource':
        if encoders is None:
            encoders = self.encoders
        name = name or self.name
        return self.__class__(name=name, tensors=tensors, length=self.length, sampling_rate=self.sampling_rate,
                              stride=self.stride, reverse=self.reverse, encoders=encoders,
                              target_params=self._target_params)

    @validate_arguments(config=ArbitraryTypes)
    def split(self, splitter: TrainValTestSpliter) -> Tuple[
        'TimeseriesDataSource', 'TimeseriesDataSource', 'TimeseriesDataSource']:
        """ For the moment it supports only OrderedSplitter, as this it TimeSeries """
        assert isinstance(splitter, OrderedSplitter)
        instance_idxes = np.arange(self.size)
        train_idxes, val_idxes, test_idxes = splitter.split(instance_idxes)
        if self._target_params is not None:
            target_raw_len = self._target_params.get_raw_target_len()
        else:
            target_raw_len = 0
        train_start, _ = self._get_window_by_instance_idx(train_idxes[0])
        _, train_end = self._get_window_by_instance_idx(train_idxes[-1])
        train_end += target_raw_len
        train_tensor = self.tensors[train_start:train_end]
        val_start, _ = self._get_window_by_instance_idx(val_idxes[0])
        _, val_end = self._get_window_by_instance_idx(val_idxes[-1])
        val_end += target_raw_len
        val_tensor = self.tensors[val_start:val_end]
        test_start, _ = self._get_window_by_instance_idx(test_idxes[0])
        _, test_end = self._get_window_by_instance_idx(test_idxes[-1])
        test_end += target_raw_len
        test_tensor = self.tensors[test_start:test_end]
        train_ds = self._clone_with_tensors_encoders(train_tensor)
        val_ds = self._clone_with_tensors_encoders(val_tensor)
        test_ds = self._clone_with_tensors_encoders(test_tensor)
        return train_ds, val_ds, test_ds

    def get_targets(self, target_name: str = 'target') -> TensorDataSource:
        """ Get the targets for the given delay and prediction length """
        if self._target_params is None:
            raise ValueError(f'This instance {self.name} of {self.__class__.__name__} was confifured without Targets')
        delay, pred_len, pred_stride = self._target_params.delay, self._target_params.pred_len, self._target_params.stride
        pred_w_sz = 1 + (pred_len - 1) * pred_stride
        non_strided_na = np.lib.stride_tricks.sliding_window_view(self.tensors[self.w_sz + delay:],
                                                                  window_shape=(pred_w_sz, self.tensors.shape[1]))
        strided_na = non_strided_na[::self.stride, 0, ::pred_stride, self._target_params.target_idx]
        if self._target_params.reverse:
            strided_na = strided_na[:, ::-1]
        assert strided_na.shape == (self.size, pred_len)
        new_data_source = TensorDataSource(name=target_name, tensors=strided_na, encoders=self.encoders)
        return new_data_source

    def __len__(self) -> int:
        return self.size

    @validate_arguments
    def _convert_negative_index(self, index: int) -> int:
        if index < 0:
            index = self.size + index
            assert 0 <= index < self.size, 'Wrong index: %s' % index
        return index

    @validate_arguments
    def _get_window_by_instance_idx(self, instance_idx: int) -> Tuple[int, int]:
        instance_idx = self._convert_negative_index(instance_idx)
        start_idx = instance_idx * self.stride
        w_sz = self.w_sz
        assert start_idx + w_sz <= self.tensors.shape[0], f'Corrupted data in generator at index {instance_idx}'
        end_idx = start_idx + w_sz
        return start_idx, end_idx

    @validate_arguments
    def _get_one_instance(self, index: int) -> np.ndarray:
        start, end = self._get_window_by_instance_idx(index)
        instance = self.tensors[start:end:self.sampling_rate]
        if self.reverse:
            return instance[::-1, ...]

        return instance

    def __getitem__(self, item: Union[int, slice, Collection[int]]) -> np.ndarray:
        if isinstance(item, int):
            return self._get_one_instance(item)
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or self.size
            step = item.step or 1
            _range = np.arange(start, stop, step)
            instances = self.get_by_idx_set(_range)
            return instances

    def get_by_idx_set(self, index_set: Collection[int]) -> np.ndarray:
        instances = np.empty((len(index_set), self.length, self.tensors.shape[-1],))
        instances.fill(np.nan)
        for batch_idx, inst_idx in enumerate(index_set):
            instances[batch_idx] = self._get_one_instance(inst_idx)
        return instances

# ForwardRef is required by PyDantic validators for self return type for Python <3.11. Python 3.11 solves this with PEP 673
TargetTimeseriesDataSource = ForwardRef('TargetTimeseriesDataSource')


class TargetTimeseriesDataSource(TimeseriesDataSource):

    @classmethod
    @validate_arguments(config=ArbitraryTypes)
    def from_timeseries_datasource(cls,
                                   ds: TimeseriesDataSource,
                                   name: Optional[str] = None
                                   ) -> 'TargetTimeseriesDataSource':
        assert ds._target_params is not None, "Can't create target datasource without target parameters"
        name = name or ds.name
        idx_arr = np.array([ds._target_params.target_idx])
        new_ds = ds.select_features(idx_arr, name)
        new_inst = cls(name=name, tensors=new_ds.tensors,
                       length=new_ds.length,
                       sampling_rate=new_ds.sampling_rate,
                       stride=new_ds.stride,
                       reverse=new_ds.reverse,
                       encoders=new_ds.encoders,
                       target_params=new_ds._target_params)
        assert len(ds) == len(new_inst)
        return new_inst

    @validate_arguments
    def _get_target_window_by_instance_idx(self, instance_idx: int) -> Tuple[int, int]:
        delay, pred_len, pred_stride = self._target_params.delay, self._target_params.pred_len, self._target_params.stride
        instance_idx = self._convert_negative_index(instance_idx)
        input_start, input_end = super()._get_window_by_instance_idx(instance_idx)
        start_idx = input_end + delay
        pred_w_sz = 1 + (pred_len - 1) * pred_stride

        assert start_idx + pred_w_sz <= self.tensors.shape[0], f'Corrupted data in generator at index {instance_idx}'
        end_idx = start_idx + pred_w_sz
        return start_idx, end_idx

    @validate_arguments
    def _get_one_instance(self, index: int) -> np.ndarray:
        start, end = self._get_target_window_by_instance_idx(index)
        stride = self._target_params.stride
        instance = self.tensors[start:end:stride, 0]
        if self._target_params.reverse:
            return instance[::-1]
        assert instance.shape == (self._target_params.pred_len,)
        return instance

    @validate_arguments(config=ArbitraryTypes)
    def get_by_idx_set(self, index_set: Collection[int]) -> np.ndarray:
        pred_len = self._target_params.pred_len
        instances = np.empty((len(index_set), pred_len))
        instances.fill(np.nan)
        for batch_idx, inst_idx in enumerate(index_set):
            instances[batch_idx] = self._get_one_instance(inst_idx)
        return instances

    def select_features(self, features: Collection[int], name: Optional[str] = None) -> DataSource:
        raise NotImplementedError("Can't implement select features on a target DataSource")

# ForwardRef is required by PyDantic validators for self return type for Python <3.11. Python 3.11 solves this with PEP 673
DataSet = ForwardRef('Foo')


class DataSet():
    @validate_arguments(config=ArbitraryTypes)
    def __init__(self,
                 input_sources: Dict[str, DataSource],
                 target_sources: Optional[Dict[str, DataSource]] = None
                 ) -> None:
        self.input_sources = input_sources
        self.target_sources = target_sources

    def _split_encode_sources_map(self,
                                  sources: Dict[str, DataSource],
                                  splitter: TrainValTestSpliter,
                                  encoders: Optional[Dict[str, List[DataEncoder]]]
                                  ) -> Tuple[Dict[str, DataSource], Dict[str, DataSource], Dict[str, DataSource]]:
        train_input_sources = {}
        val_input_sources = {}
        test_input_sources = {}
        if encoders is None:
            encoders = defaultdict(list)
        for name, data_source in sources.items():
            train, val, test = data_source.split(splitter)
            encoded_train = train.fit_encode(encoders[name])
            train_input_sources[name] = encoded_train
            val_input_sources[name] = val.encode(encoded_train.get_encoders())
            test_input_sources[name] = test.encode(encoded_train.get_encoders())
        return train_input_sources, val_input_sources, test_input_sources

    @validate_arguments(config=ArbitraryTypes)
    def split(self, splitter: TrainValTestSpliter) -> Tuple['DataSet', 'DataSet', 'DataSet']:
        return self.split_encode(splitter, None)

    @validate_arguments(config=ArbitraryTypes)
    def split_encode(self, splitter: TrainValTestSpliter, encoders: Optional[Dict[str, List[DataEncoder]]]) \
            -> Tuple['DataSet', 'DataSet', 'DataSet']:
        """
        The splitter has to be reproducible, i.e. don't use RandomSplitter without setting the seed.
        Note: you can instantiate RandomSplitter with randomly runtime-selected seed.
        """
        assert splitter.is_reproducible()
        train_inputs, val_inputs, test_inputs = self._split_encode_sources_map(self.input_sources,
                                                                               splitter, encoders)
        if self.target_sources:
            train_targets, val_targets, test_targets = self._split_encode_sources_map(self.target_sources,
                                                                                      splitter, encoders)
        else:
            train_targets = val_targets = test_targets = None
        train_ds = DataSet(train_inputs, train_targets)
        val_ds = DataSet(val_inputs, val_targets)
        test_ds = DataSet(test_inputs, test_targets)
        return train_ds, val_ds, test_ds

    def get_encoders(self) -> Dict[str, List[DataEncoder]]:
        both_sources = (self.input_sources, self.target_sources)
        all_encoders = {name: source.get_encoders() for sources in both_sources for name, source in sources.items()}
        return all_encoders


class XBatchGenerator(data_utils.Sequence):
    @validate_arguments(config=ArbitraryTypes)
    def __init__(self,
                 inputs_map: Dict[str, DataSource],
                 batch_size: PositiveInt = 128
                 ):
        self.inputs = inputs_map
        first_ds = inputs_map[list(inputs_map.keys())[0]]
        self.n_instances = len(first_ds)
        self.n_batches = math.ceil(self.n_instances / batch_size)
        for k, input_seq in inputs_map.items():
            assert self.n_instances == len(input_seq), f"{k} input shape mismatch"
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.n_batches

    @validate_arguments
    def _get_batch(self, start_index: int, end_index: int) -> Dict[str, np.ndarray]:
        inputs = {}
        for k, input_seq in self.inputs.items():
            inputs[k] = input_seq[start_index:end_index]
        return inputs

    def __getitem__(self, index):
        if index >= self.n_batches:
            raise IndexError(f"Index {index} is out of range")
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.n_instances)
        return self._get_batch(start_index, end_index)


class XYBatchGenerator(XBatchGenerator):
    @validate_arguments(config=ArbitraryTypes)
    def __init__(self,
                 inputs_map: Dict[str, DataSource],
                 targets_map: Dict[str, DataSource],
                 shuffle: bool = True,
                 batch_size: int = 128
                 ):
        super().__init__(inputs_map, batch_size)
        self.targets = targets_map
        first_ds = inputs_map[list(inputs_map.keys())[0]]
        self.n_instances = len(first_ds)
        self.n_batches = math.ceil(self.n_instances / batch_size)
        for k, input_seq in inputs_map.items():
            assert self.n_instances == len(input_seq), f"{k} input shape mismatch"
        for k, target_seq in targets_map.items():
            assert self.n_instances == len(target_seq), f"{k} input shape mismatch"

        self.shuffle = shuffle
        if shuffle:
            self._shuffle_index = np.arange(self.n_instances)
            np.random.shuffle(self._shuffle_index)
        self.on_epoch_end()

    @validate_arguments
    def _get_batch(self, start_index: int, end_index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        inputs = {}
        rows = None
        if self.shuffle:
            rows = self._shuffle_index[start_index:end_index]

        for k, input_seq in self.inputs.items():
            if self.shuffle:
                inputs[k] = input_seq.get_by_idx_set(rows)
            else:
                inputs[k] = input_seq[start_index:end_index]
        targets = {}
        for k, target_seq in self.targets.items():
            if self.shuffle:
                targets[k] = target_seq.get_by_idx_set(rows)
            else:
                targets[k] = target_seq[start_index:end_index]
        return inputs, targets

    def get_config(self):
        """Returns the configuration of the generator.
        A generator's configuration is the keyword arguments
        given to `__init__`.

        # Returns
            A dictionary.
        """
        return {
            'inputs':        json.dumps(self.inputs, cls=NumpyArrayEncoder),
            'targets':       json.dumps(self.targets, cls=NumpyArrayEncoder),
            'sampling_rate': self.sampling_rate,
            'shuffle':       self.shuffle,
            'batch_size':    self.batch_size
        }

    def to_json(self, **kwargs):
        """Returns a JSON string containing the timeseries generator
        configuration. To load a generator from a JSON string, use
        `keras.preprocessing.sequence.timeseries_generator_from_json(json_string)`.

        # Arguments
            **kwargs: Additional keyword arguments
                to be passed to `json.dumps()`.

        # Returns
            A JSON string containing the tokenizer configuration.
        """
        config = self.get_config()
        timeseries_generator_config = {
            'class_name': self.__class__.__name__,
            'config':     config
        }
        return json.dumps(timeseries_generator_config, **kwargs)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._shuffle_index)
        super().on_epoch_end()

    def __add__(self, other: 'XYBatchGenerator') -> 'XYBatchGenerator':
        inputs = {k: v + other.inputs[k] for k, v in self.inputs.items()}
        targets = {k: v + other.targets[k] for k, v in self.targets.items()}
        return self.__class__(inputs, targets, self.shuffle, self.batch_size, self.sampling_rate)

    def get_X_generator(self, batch_size: Optional[int] = None) -> XBatchGenerator:
        batch_size = batch_size or self.batch_size
        return XBatchGenerator(self.inputs, batch_size)
