#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

import gzip
import logging
import pickle
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import tensorflow as tf
from keras import Model
from keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    History,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tsx import TS

from ..callbacks import MetricCheckpoint
from ..encoders import DataEncoder
from ..generators import DataSource, TensorDataSource, XBatchGenerator, XYBatchGenerator
from ..model_abstractions.model_params import ModelParams


class ModelObject(ABC):
    def __init__(
        self,
        mp: ModelParams,
        model: Model,
        encoders: Dict[str, List[DataEncoder]],
    ) -> None:
        self.mp = mp
        self.model = model
        self.encoders = encoders

    def train(
        self,
        train_gen: XYBatchGenerator,
        val_gen: XYBatchGenerator,
        device: str = "/CPU:0",
        callbacks: Union[List[Callback], Tuple[Callback, ...]] = (),
        verbose=1,
    ) -> History:
        mp = self.mp

        with tf.device(device):
            logging.info(f"Starting training with mp: {mp!s}")
            history = self.model.fit(
                train_gen,
                epochs=mp.max_epochs,
                callbacks=callbacks,
                validation_data=val_gen,
                verbose=verbose,
            )
        return history

    def predict(self, x: XBatchGenerator, device: str = "/CPU:0") -> DataSource:
        """
        Note: the XBatchGenerator must be created with the same encoders as the ones used to train the model.
        """
        with tf.device(device):
            res = self.model.predict(x)
        res_ds = TensorDataSource(name="prediction", tensors=res, encoders=self.encoders[self.mp.target_name])
        decoded = res_ds.decode()
        return decoded

    def predict_raw(self, raw_input_sources: Dict[str, DataSource], device: str = "/CPU:0") -> DataSource:
        """
        This method is used to predict on raw input sources - i.e. unscaled/unnormalized.
        """
        scaled_input_tds = {name: tds.encode(self.encoders[name]) for name, tds in raw_input_sources.items()}
        x_gen = XBatchGenerator(scaled_input_tds)
        return self.predict(x_gen, device=device)

    def evaluate_raw(self, xy: XYBatchGenerator, device: str = "/CPU:0") -> Dict[str, float]:
        """
        This method is used to evaluate on raw input sources - i.e. unscaled/unnormalized.
        """
        scaled_inputs = {name: tds.encode(self.encoders[name]) for name, tds in xy.inputs.items()}
        scaled_targets = {name: tds.encode(self.encoders[name]) for name, tds in xy.targets.items()}
        scaled_xy = XYBatchGenerator(scaled_inputs, scaled_targets, shuffle=False, batch_size=xy.batch_size)
        with tf.device(device):
            res = self.model.evaluate(scaled_xy)
        return {self.model.metrics_names[i]: v for i, v in enumerate(res)}

    @staticmethod
    def construct_model_dir(name: str, base_dir: Union[str, Path] = "model-data") -> Path:
        model_save_name = f'{TS.now().as_file_ts}-{name}'
        model_dir = (Path(base_dir).absolute()) / model_save_name
        return model_dir


class SimpleModelObject(ModelObject):
    def train(
        self,
        train_gen: XYBatchGenerator,
        val_gen: XYBatchGenerator,
        device: str = "/CPU:0",
        callbacks: Union[List[Callback], Tuple[Callback, ...]] = (),
        verbose=1,
        model_dir: Optional[Path] = None,
        add_default_callbacks: bool = True,
    ) -> History:
        model_dir.mkdir(parents=True, exist_ok=True)
        mp = self.mp
        _callbacks = list(callbacks)
        if add_default_callbacks:
            if mp.rop_patience is not None:
                rop = ReduceLROnPlateau(
                    factor=0.5,
                    patience=mp.rop_patience,
                    verbose=1,
                    cooldown=0,
                    min_lr=5e-05,
                )
                _callbacks.append(rop)
            if mp.early_stop_patience is not None:
                early_stopping = EarlyStopping(patience=mp.early_stop_patience, min_delta=0.0001)
                _callbacks.append(early_stopping)

            metrics_checkpoint = MetricCheckpoint(model_dir)
            _callbacks.append(metrics_checkpoint)

            csv_logger = CSVLogger(filename=str(model_dir / "metrics.csv"), append=True)
            _callbacks.append(csv_logger)

            path_pattern = model_dir / "weights_e{epoch:03d}_tl{loss:.8f}_vl{val_loss:.8f}.hdf5"
            checkpoint = ModelCheckpoint(
                str(path_pattern),
                monitor="val_loss",
                verbose=1,
                save_best_only=False,
                save_weights_only=False,
            )
            _callbacks.append(checkpoint)

        mp.serialize_to_file(model_dir / "mp.json")
        self.save_scalers(model_dir)
        with tf.device(device):
            logging.info(f"Starting training with mp: {mp!s}")
            try:
                history = self.model.fit(
                    train_gen,
                    epochs=mp.max_epochs,
                    callbacks=_callbacks,
                    validation_data=val_gen,
                    steps_per_epoch=mp.steps_per_epoch,
                    verbose=verbose,
                )
            except Exception:  # pylint: disable=broad-except
                if _callbacks:
                    # Next callbacks do not close their files in case the train fails
                    for cb in _callbacks:
                        try:
                            cb.on_train_end(None)
                        except Exception:  # pylint: disable=broad-except
                            pass
                    raise
        return history

    def save_scalers(self, model_dir: Path):
        encoders_dir = model_dir / "encoders"
        encoders_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(encoders_dir / "encoders.pk.gz", "wb") as f:
            pickle.dump(self.encoders, f)

    @classmethod
    def from_model_dir(
        cls,
        hdf5_path: Path,
        model_params_cls: Type[ModelParams],
        device: str = "/CPU:0",
        custom_classes: Optional[List[Any]] = None,
    ):
        model_dir = hdf5_path.parent
        mp = model_params_cls.from_file(model_dir / "mp.json")
        encoders_dir = model_dir / "encoders"
        with gzip.open(encoders_dir / "encoders.pk.gz", "rb") as f:
            encoders = pickle.load(f)

        with tf.device(device):
            custom_object_classes = [model_params_cls] + (custom_classes or [])
            custom_objects = {cls.__name__: cls for cls in custom_object_classes}
            model = tf.keras.models.load_model(hdf5_path, custom_objects=custom_objects)
        return cls(mp=mp, model=model, encoders=encoders)
