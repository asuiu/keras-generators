#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

from json import JSONEncoder

import numpy as np
from pydantic import Extra


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class ImmutableConfig:
    allow_mutation = False
    arbitrary_types_allowed = True
    extra = Extra.forbid


class ArbitraryTypes:
    arbitrary_types_allowed = True
