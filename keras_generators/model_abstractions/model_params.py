#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

import json
from pathlib import Path
from typing import Optional, List

from pydantic import constr, conint, PositiveInt, confloat
from pyxtension.models import ImmutableExtModel
from typing_extensions import Literal


class ModelParams(ImmutableExtModel):
    """
    This class is a base class used to encode the parameters of a model.
    It's easy serializable and deserializable to/from a file.
    It's immutable, and versioned. It won't deserialize an instance of a class with a different version.
    Feel free to extend this class and add more hyper-parameters and architecture parameters for your model like:
        - number of Dense layers
        - type of RNN layer (LSTM, GRU or RNN)
        - add different versions of the ModelParams
        - etc.
    """
    version: constr(regex=r"^v1\.0\.0$", strict=True) = "v1.0.0"

    batch_size: conint(ge=-1) = 64
    val_batch_size: PositiveInt = 64
    train_size_ratio: confloat(ge=0, le=1) = 0.6
    val_size_ratio: confloat(ge=0, le=1) = 0.2
    split_policy: Literal["random", "sequential"] = 'random'  # The policy for splitting dataset

    # Maximum number of epochs to train the model
    max_epochs: PositiveInt = 100

    # Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
    steps_per_epoch: Optional[int] = None
    early_stop_patience: Optional[
        PositiveInt] = None  # The number of epochs with no improvement after which training will be stopped
    rop_patience: Optional[PositiveInt] = None  # Reduce on plateau patience
    learning_rate: confloat(ge=0, le=1) = 0.003

    loss: Literal['mse'] = 'mse'
    metrics: List[Literal['mse']] = ['mse']

    input_name: str = "input"  # Name of the main input layer
    target_name: str = "target"  # Name of the main target layer

    def get_config(self):
        """Returns the config dictionary for a `ModelParams` instance."""
        return {"data": self.json()}

    def __repr__(self) -> str:
        str_repr = self.__repr_str__(",\n")
        return f"{self.__repr_name__()}({str_repr})"

    @property
    def out_dim(self):
        """
        Represents the size of the output vector/nr neurons (in case it's one single vector).
        """
        return 1

    def serialize_to_file(self, fpath: Path):
        # ToDo: use a better json library
        # Because the Python json library is buggy and doesn't respect custom encoder, we have to convert TS type
        # manually details: https://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module/5574182
        # Fixed this by overriding BaseModel.json() method and use custom encoder
        with fpath.open("wt") as f:
            f.write(self.json(indent=2))

    @classmethod
    def from_config(cls, config):
        d = json.loads(config["data"])
        return cls(**d)

    @classmethod
    def from_file(cls, fpath: Path) -> 'ModelParams':
        with fpath.open("rt") as f:
            d = json.load(f)
            mp = cls(**d)
            return mp
