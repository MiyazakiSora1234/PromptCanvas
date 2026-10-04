"""HTTP layer: routes under ``/api``, request validation, error responses, queueing."""

from .routes import router

__all__ = ["router"]
