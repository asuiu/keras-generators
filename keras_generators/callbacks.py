import json
import logging
import math
from json import JSONEncoder
from pathlib import Path
from typing import Optional

import dill
import numpy as np
from tensorflow.python.util.tf_export import keras_export
from tf_keras.callbacks import Callback, CSVLogger, ReduceLROnPlateau, TensorBoard
from typing_extensions import Self, override

from keras_generators.common import SerializableCallback


class EarlyStoppingAtMinLoss(Callback, SerializableCallback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, loss_max_diff: float, patience=2, min_delta=0.0001):
        super().__init__()

        self.patience = patience
        assert min_delta > 0.0
        self.min_delta = min_delta

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.monitor = "val_loss"
        self.loss_max_diff = loss_max_diff
        self.wait: Optional[int] = None
        self.stopped_epoch: Optional[int] = None
        self.best: Optional[int] = None

    def on_train_begin(self, *_a, **_k):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def _stop_train(self, epoch, message: str = ""):
        self.stopped_epoch = epoch
        self.model.stop_training = True
        logging.info("Restoring model weights from the end of the best epoch. %s", message)
        logging.warning(message)
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        train_loss = logs.get("loss")
        if current is None:
            print(f"No val_loss yet: {logs!s}")
            return
        if current - train_loss > self.loss_max_diff:
            self._stop_train(epoch, f"Stopping because loss difference {current - train_loss}")
        if math.isnan(current):
            self._stop_train(epoch)
        if np.less(current, self.best):
            # Record the best weights if current results is better (less).
            self._improved()
        if np.less(current, self.best - self.min_delta):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._stop_train(epoch)

    def _improved(self):
        self.best_weights = self.model.get_weights()

    def on_train_end(self, *_a, **_k):
        if self.stopped_epoch > 0:
            logging.warning("Epoch %05d: early stopping", self.stopped_epoch + 1)


class MetricCheckpoint(Callback, SerializableCallback):
    FNAME = "metrics.jsonl"

    def on_train_begin(self, *_args, **_kwargs):
        self._f = open(self._dir / self.FNAME, "at", encoding="utf-8")  # pylint: disable=consider-using-with

    def __init__(self, model_dir: Path):
        super().__init__()
        self._f = None
        self._dir: Path = model_dir
        self._je = JSONEncoder()

    def to_json_obj(self, obj):
        try:
            self._je.encode(obj)
        except TypeError:
            return float(obj)
        return obj

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            log = {k: self.to_json_obj(v) for k, v in logs.items()}
            log["_epoch"] = epoch
            self._f.write(json.dumps(log) + "\n")
            self._f.flush()

    def on_train_end(self, *_args, **_kwargs):
        self._f.close()


@keras_export("keras.callbacks.SerializableReduceLROnPlateau")
class SerializableReduceLROnPlateau(ReduceLROnPlateau, SerializableCallback):
    pass


@keras_export("keras.callbacks.SerializableTensorBoard")
class SerializableTensorBoard(TensorBoard, SerializableCallback):
    @override
    def serialize(self) -> bytes:
        kwargs = {
            "log_dir": self.log_dir,
            "histogram_freq": self.histogram_freq,
            "write_graph": self.write_graph,
            "write_images": self.write_images,
            "write_steps_per_second": self.write_steps_per_second,
            "update_freq": self.update_freq,
            # "profile_batch" we ignore this parameter as it's not directly saved on the object
            "embeddings_freq": self.embeddings_freq,
            "embeddings_metadata": self.embeddings_metadata,
        }
        all_state_attrs = self.__dict__.copy()
        excepted_attrs = set(kwargs.keys()) | {"_writers", "model"}
        for attr in excepted_attrs:
            del all_state_attrs[attr]

        return dill.dumps((kwargs, all_state_attrs))

    @override
    @classmethod
    def deserialize(cls, buffer: bytes) -> Self:
        kwargs, state_attrs = dill.loads(buffer)
        instance = cls(**kwargs)
        for attr, val in state_attrs.items():
            setattr(instance, attr, val)
        return instance


@keras_export("keras.callbacks.SerializableCSVLogger")
class SerializableCSVLogger(CSVLogger, SerializableCallback):
    def serialize(self) -> bytes:
        """
        self.sep = separator
        self.filename = io_utils.path_to_string(filename)
        self.append = append

        :return:
        """
        kwargs = {
            "separator": self.sep,
            "filename": self.filename,
            "append": self.append,
        }
        all_state_attrs = self.__dict__.copy()
        excepted_attrs = {"sep", "filename", "append"} | {"writer", "csv_file", "model"}
        for attr in excepted_attrs:
            del all_state_attrs[attr]

        return dill.dumps((kwargs, all_state_attrs))

    @override
    @classmethod
    def deserialize(cls, buffer: bytes) -> Self:
        kwargs, state_attrs = dill.loads(buffer)
        instance = cls(**kwargs)
        for attr, val in state_attrs.items():
            setattr(instance, attr, val)
        return instance
