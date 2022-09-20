#!/usr/bin/env python
# Author: ASU --<andrei.suiu@gmail.com>

import json
from typing import Any, Callable, Optional, cast

from pydantic import BaseModel, Extra
from tsx import TS


class TSJSONEncoder(json.JSONEncoder):
    def encode(self, o: Any) -> str:
        if isinstance(o, dict):
            new_o = o.copy()
            for k in new_o:
                if isinstance(new_o[k], TS):
                    new_o[k] = new_o[k].as_iso
            return super().encode(new_o)
        return super().encode(o)


class SerializableStruct(BaseModel):
    def json(
            self,
            *,
            include=None,
            exclude=None,
            by_alias: bool = False,
            skip_defaults: bool = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            encoder: Optional[Callable[[Any], Any]] = None,
            **dumps_kwargs: Any,
    ) -> str:
        """
        Generate a JSON representation of the model, `include` and `exclude` arguments as per `dict()`.

        `encoder` is an optional function to supply as `default` to json.dumps(), other arguments as per `json.dumps()`.
        """
        if skip_defaults is not None:
            exclude_unset = skip_defaults
        encoder = cast(Callable[[Any], Any], encoder or self.__json_encoder__)
        data = self.dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        if self.__custom_root_type__:
            # below is a hardcoding workaround instead of original utils.ROOT_KEY as Pydantic doesn't have it on Unix
            data = data["__root__"]

        return self.__config__.json_dumps(data, default=encoder, cls=TSJSONEncoder, **dumps_kwargs)

    class Config:
        arbitrary_types_allowed = True


class SerializableImmutableStruct(SerializableStruct):
    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        extra = Extra.forbid
