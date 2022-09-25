#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

import gzip
import logging
import pickle
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional, Type, Any

import tensorflow as tf
from keras import Model
from keras.callbacks import Callback, History, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger

from ..encoders import DataEncoder
from ..generators import XYBatchGenerator, XBatchGenerator, TensorDataSource, DataSource
from ..model_abstractions.model_params import ModelParams


class ModelObject(ABC):

    def __init__(self,
                 mp: ModelParams,
                 model: Model,
                 encoders: Dict[str, List[DataEncoder]],
                 ) -> None:
        self.mp = mp
        self.model = model
        self.encoders = encoders

    def train(self,
              train_gen: XYBatchGenerator,
              val_gen: XYBatchGenerator,
              device: str = "/CPU:0",
              callbacks: Union[List[Callback], Tuple[Callback, ...]] = (),
              verbose=1
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

    def predict(self, x: XBatchGenerator,
                device: str = "/CPU:0"
                ) -> DataSource:
        with tf.device(device):
            res = self.model.predict(x)
        res_ds = TensorDataSource(name="prediction", tensors=res, encoders=self.encoders[self.mp.target_name])
        decoded = res_ds.decode()
        return decoded

    @staticmethod
    def construct_model_dir(name: str, base_dir: Union[str, Path] = "model-data") -> Path:
        model_save_name = f'{name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        model_dir = (Path(base_dir).absolute()) / model_save_name
        return model_dir


class SimpleModelObject(ModelObject):

    def __init__(self,
                 mp: ModelParams,
                 model: Model,
                 encoders: Dict[str, List[DataEncoder]],
                 ) -> None:
        super().__init__(mp, model, encoders)

    def train(self,
              train_gen: XYBatchGenerator,
              val_gen: XYBatchGenerator,
              device: str = "/CPU:0",
              callbacks: Union[List[Callback], Tuple[Callback, ...]] = (),
              model_dir: Optional[Path] = None,
              verbose=1,
              ) -> History:
        model_dir.mkdir(parents=True, exist_ok=True)
        mp = self.mp
        _callbacks = list(callbacks)
        if mp.rop_patience is not None:
            rop = ReduceLROnPlateau(factor=0.5, patience=mp.rop_patience, verbose=1, cooldown=0, min_lr=5e-05)
            _callbacks.append(rop)
        if mp.early_stop_patience is not None:
            early_stopping = EarlyStopping(patience=mp.early_stop_patience, min_delta=0.0001)
            _callbacks.append(early_stopping)

        mp.serialize_to_file(model_dir / f"mp.json")
        self.save_scalers(model_dir)

        path_pattern = model_dir / "weights_e{epoch:03d}_tl{loss:.8f}_vl{val_loss:.8f}.hdf5"
        checkpoint = ModelCheckpoint(
            str(path_pattern), monitor="val_loss", verbose=1, save_best_only=False, save_weights_only=False
        )
        _callbacks.append(checkpoint)
        csv_logger = CSVLogger(filename=str(model_dir / "metrics.csv"), append=True)
        _callbacks.append(csv_logger)

        try:
            with tf.device(device):
                logging.info(f"Starting training with mp: {mp!s}")
                history = self.model.fit(
                    train_gen,
                    epochs=mp.max_epochs,
                    callbacks=_callbacks,
                    validation_data=val_gen,
                    steps_per_epoch=mp.steps_per_epoch,
                    verbose=verbose,
                )
        except Exception:
            # Next callbacks do not close their files in case the train fails
            for cb in [csv_logger]:
                try:
                    cb.on_train_end(None)
                except Exception:
                    pass
            raise
        return history

    def save_scalers(self, model_dir: Path):
        encoders_dir = model_dir / 'encoders'
        encoders_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(encoders_dir / f'encoders.pk.gz', "wb") as f:
            pickle.dump(self.encoders, f)

    @classmethod
    def from_model_dir(cls,
                       hdf5_path: Path,
                       model_params_cls: Type[ModelParams],
                       device: str = "/CPU:0",
                       custom_classes: Optional[List[Any]] = None
                       ) -> 'ModelObject':
        model_dir = hdf5_path.parent
        mp = model_params_cls.from_file(model_dir / "mp.json")
        encoders_dir = model_dir / 'encoders'
        with gzip.open(encoders_dir / f'encoders.pk.gz', "rb") as f:
            encoders = pickle.load(f)

        with tf.device(device):
            custom_object_classes = [model_params_cls] + (custom_classes or [])
            custom_objects = {cls.__name__: cls for cls in custom_object_classes}
            model = tf.keras.models.load_model(hdf5_path, custom_objects=custom_objects)
        return cls(mp=mp, model=model, encoders=encoders)
