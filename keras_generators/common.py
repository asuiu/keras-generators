#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

from json import JSONEncoder

import dill
import numpy as np
from pydantic import Extra


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


class SerializableKerasObject:
    def serialize(self) -> bytes:
        pickled = dill.dumps(self)
        return pickled

    @classmethod
    def deserialize(cls, buffer: bytes) -> "SerializableKerasObject":
        return dill.loads(buffer)
