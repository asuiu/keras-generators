import json
import logging
import math
import pickle
from json import JSONEncoder
from pathlib import Path
from typing import Optional

import numpy as np
from keras.callbacks import Callback, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.python.util.tf_export import keras_export


class EarlyStoppingAtMinLoss(Callback):
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
        logging.info(
            "Restoring model weights from the end of the best epoch. %s", message
        )
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
            self._stop_train(
                epoch, f"Stopping because loss difference {current - train_loss}"
            )
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
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class MetricCheckpoint(Callback):
    FNAME = "metrics.jsonl"

    def on_train_begin(self, *_args, **_kwargs):
        self._f = open(
            self._dir / self.FNAME, "at"
        )  # pylint: disable=consider-using-with

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
        else:
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
class SerializableReduceLROnPlateau(ReduceLROnPlateau):
    def serialize(self) -> bytes:
        kwargs = {
            "monitor": self.monitor,
            "factor": self.factor,
            "patience": self.patience,
            "verbose": self.verbose,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "cooldown": self.cooldown,
            "min_lr": self.min_lr,
        }
        state_attrs = {
            "_chief_worker_only": self._chief_worker_only,
            "_keras_api_names": self._keras_api_names,
            "_keras_api_names_v1": self._keras_api_names_v1,
            "_supports_tf_logs": self._supports_tf_logs,
            "wait": self.wait,
            "cooldown_counter": self.cooldown_counter,
            "best": self.best,
            "params": self.params,
            "validation_data": self.validation_data,
        }

        return pickle.dumps((kwargs, state_attrs))

    @classmethod
    def deserialize(cls, buffer: bytes) -> "SerializableReduceLROnPlateau":
        kwargs, state_attrs = pickle.loads(buffer)
        instance = cls(**kwargs)
        for attr, val in state_attrs.items():
            setattr(instance, attr, val)
        return instance


@keras_export("keras.callbacks.SerializableTensorBoard")
class SerializableTensorBoard(TensorBoard):
    def serialize(self) -> bytes:
        kwargs = {
            "log_dir": self.log_dir,
            "histogram_freq": self.histogram_freq,
            "write_graph": self.write_graph,
            "write_images": self.write_images,
            "update_freq": self.update_freq,
            "embeddings_freq": self.embeddings_freq,
            "embeddings_metadata": self.embeddings_metadata,
        }

        all_state_attrs = self.__dict__.copy()
        excepted_attrs = set(kwargs.keys()) | {"_writers", "model"}
        for attr in excepted_attrs:
            del all_state_attrs[attr]

        return pickle.dumps((kwargs, all_state_attrs))

    @classmethod
    def deserialize(cls, buffer: bytes) -> "SerializableTensorBoard":
        kwargs, state_attrs = pickle.loads(buffer)
        instance = cls(**kwargs)
        for attr, val in state_attrs.items():
            setattr(instance, attr, val)
        return instance


@keras_export("keras.callbacks.SerializableCSVLogger")
class SerializableCSVLogger(CSVLogger):
    def serialize(self) -> bytes:
        kwargs = {"filename": self.filename}

        all_state_attrs = self.__dict__.copy()
        # writer and csv_file are layzily initialized by object
        # model is set before training, AFAIK
        excepted_attrs = set(kwargs.keys()) | {"model", "writer", "csv_file"}
        for attr in excepted_attrs:
            del all_state_attrs[attr]

        return pickle.dumps((kwargs, all_state_attrs))

    @classmethod
    def deserialize(cls, buffer: bytes) -> "SerializableCSVLogger":
        kwargs, state_attrs = pickle.loads(buffer)
        instance = cls(**kwargs)
        for attr, val in state_attrs.items():
            setattr(instance, attr, val)
        return instance
