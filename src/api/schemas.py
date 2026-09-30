"""Request models, generated from the dataclasses they mirror so the two cannot drift."""
import dataclasses

from pydantic import ConfigDict, Field, create_model

from src.tools.price_return import Params


def _model_from_dataclass(cls, name):
    """A pydantic model with the dataclass's fields, types and defaults; unknown keys rejected."""
    definitions = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            annotation = f.type if f.default is not None else f.type | None
            definitions[f.name] = (annotation, f.default)
        else:
            definitions[f.name] = (f.type, Field(default_factory=f.default_factory))
    return create_model(name, __config__=ConfigDict(extra='forbid'), **definitions)


ParamsIn = _model_from_dataclass(Params, 'ParamsIn')
ParamsIn.__doc__ = 'Every `Params` field, with the same defaults. Unknown keys are rejected.'


def to_params(body):
    """The `Params` a request describes."""
    return Params(**body.model_dump())
