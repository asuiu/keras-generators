#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

from json import JSONEncoder
from typing import Type

import dill
import numpy as np
from pydantic import Extra
from tf_keras.src.callbacks import Callback


class NumpyArrayEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)


class ImmutableConfig:
    allow_mutation = False
    arbitrary_types_allowed = True
    extra = Extra.forbid


class ArbitraryTypes:
    arbitrary_types_allowed = True


class SerializableCallback:
    def serialize(self: Callback) -> bytes:
        """
        Attention: This method is not thread-safe. It temporary sets the self.model to None in order to serialize the object.

        The setting of the model is done in order to avoid pickling the model, which takes redundant space,
            as well as has multiple issues on deserialization.
        """
        model = self.model  # pylint: disable=access-member-before-definition
        self.model = None  # pylint: disable=attribute-defined-outside-init
        pickled = dill.dumps(self)
        self.model = model  # pylint: disable=attribute-defined-outside-init
        return pickled

    @classmethod
    def deserialize(cls: Type[Callback], buffer: bytes) -> "SerializableCallback":
        keras_object = dill.loads(buffer)
        return keras_object
