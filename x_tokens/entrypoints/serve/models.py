"""Served-model lookup without exposing Core model objects."""

from __future__ import annotations

from .errors import OpenAIError


class ModelRegistry:
    def __init__(self, served_model_name: str) -> None:
        self._served_model_name = served_model_name

    def validate(self, model: str) -> None:
        if model != self._served_model_name:
            raise OpenAIError(
                404,
                "The requested model does not exist",
                "invalid_request_error",
                "model",
                "model_not_found",
            )

    def list_models(self) -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self._served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "x-tokens",
                }
            ],
        }
