"""Utility functions"""

import json
from typing import TypeVar
from pydantic import BaseModel, ValidationError


T = TypeVar('T', bound=BaseModel)


def try_parse_to_model(json_string: str, model: type[T]) -> T | None:
    """Check if a dictionary is a valid model."""
    try:
        dictionary = json.loads(json_string)
    except json.JSONDecodeError:
        return None
    try:
        return model.model_validate(dictionary)
    except ValidationError:
        return None
