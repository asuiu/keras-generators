import json
import math
from json import JSONEncoder
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.random import MT19937
from keras.utils import data_utils

from .dtypes import ScaledTensorSeq, Scaler, TensorSeq
from .splitters import TrainValTestSpliter


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class XBatchGenerator(data_utils.Sequence):
    def __init__(self,
                 inputs_map: Dict[str, ScaledTensorSeq],
                 batch_size: int = 128
                 ):
        self.inputs = inputs_map
        self.n_batches = math.ceil(len(inputs_map[list(inputs_map.keys())[0]].norm_na) / batch_size)
        self.n_instances = len(inputs_map[list(inputs_map.keys())[0]].norm_na)
        input_shapes = []
        for k, input_seq in inputs_map.items():
            input_shapes.append(len(input_seq.norm_na))
        for shape in input_shapes[1:]:
            assert shape == input_shapes[0], f"{shape} input shape mismatch"
        input_shapes = []
        for shape in input_shapes[1:]:
            assert shape == input_shapes[0], f"{shape} target shape mismatch"
        self.batch_size = batch_size

    def __len__(self):
        return self.n_batches

    def _get_batch(self, start_index: int, end_index: int) -> Dict[str, np.ndarray]:
        inputs = {}
        for k, input_seq in self.inputs.items():
            inputs[k] = input_seq.norm_na[start_index:end_index]
        return inputs

    def __getitem__(self, index):
        if index >= self.n_batches:
            raise IndexError(f"Index {index} is out of range")
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.n_instances)
        return self._get_batch(start_index, end_index)


class XYBatchGenerator(XBatchGenerator):
    def __init__(self,
                 inputs_map: Dict[str, ScaledTensorSeq],
                 targets_map: Dict[str, ScaledTensorSeq],
                 shuffle: bool = True,
                 batch_size: int = 128
                 ):
        super().__init__(inputs_map, batch_size)
        self.targets = targets_map
        output_shapes = []
        for k, target_seq in targets_map.items():
            output_shapes.append(len(target_seq.norm_na))
        for shape in output_shapes[1:]:
            assert shape == output_shapes[0], f"{shape} target shape mismatch"
        self.shuffle = shuffle
        if shuffle:
            self._shuffle_index = np.arange(self.n_instances)
            np.random.shuffle(self._shuffle_index)
        self.on_epoch_end()

    def _get_batch(self, start_index: int, end_index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        inputs = {}
        rows = None
        if self.shuffle:
            rows = self._shuffle_index[start_index:end_index]

        for k, input_seq in self.inputs.items():
            if self.shuffle:
                inputs[k] = input_seq.norm_na[rows]
            else:
                inputs[k] = input_seq.norm_na[start_index:end_index]
        targets = {}
        for k, target_seq in self.targets.items():
            if self.shuffle:
                targets[k] = target_seq.norm_na[rows]
            else:
                targets[k] = target_seq.norm_na[start_index:end_index]
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


class TrainData:
    def __init__(self,
                 inputs: Dict[str, TensorSeq],
                 targets: Dict[str, TensorSeq],
                 splitter: TrainValTestSpliter,
                 train_batch_size: int,
                 val_batch_size: int,
                 scaler_map: Dict[str, Optional[Scaler]],
                 shuffle: bool = True,
                 reshuffle=True
                 ):
        self._shuffle = shuffle
        self._reshuffle = reshuffle

        self.train_inputs = {}
        self.val_inputs = {}
        self.test_inputs = {}
        self.scaler_map = scaler_map
        for name, numeric_tensor in inputs.items():
            train_seq, val_seq, test_seq = numeric_tensor.split(splitter)
            scaler = scaler_map[name]
            assert not hasattr(scaler, 'scale_')
            self.train_inputs[name] = train_seq.scale(scaler)
            self.val_inputs[name] = val_seq.scale(scaler)
            self.test_inputs[name] = test_seq.scale(scaler)

        self.train_targets = {}
        self.val_targets = {}
        self.test_targets = {}
        for name, numeric_tensor in targets.items():
            train_seq, val_seq, test_seq = numeric_tensor.split(splitter)
            scaler = scaler_map[name]
            assert scaler is None or not hasattr(scaler, 'scale_')
            self.train_targets[name] = train_seq.scale(scaler)
            assert scaler is None or hasattr(scaler, 'scale_')
            self.val_targets[name] = val_seq.scale(scaler)
            self.test_targets[name] = test_seq.scale(scaler)

        self.train_gen = XYBatchGenerator(self.train_inputs, self.train_targets, batch_size=train_batch_size,
                                          shuffle=self._shuffle)
        self.val_gen = XYBatchGenerator(self.val_inputs, self.val_targets, batch_size=val_batch_size,
                                        shuffle=self._shuffle)
        self.test_gen = XYBatchGenerator(self.test_inputs, self.test_targets, batch_size=val_batch_size,
                                         shuffle=self._shuffle)

    def get_scalers(self) -> Dict[str, Scaler]:
        return self.scaler_map


class InferenceData:
    def __init__(self,
                 inputs: Dict[str, TensorSeq],
                 batch_size: int,
                 scaler_map: Dict[str, Optional[Scaler]]
                 ) -> None:
        self.inputs = {k: v.scale(scaler_map[k]) for k, v in inputs.items()}
        self.gen = XBatchGenerator(self.inputs, batch_size=batch_size)
