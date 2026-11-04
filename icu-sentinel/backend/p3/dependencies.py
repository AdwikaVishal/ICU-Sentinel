"""FastAPI dependencies for the P3 API."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from backend.p3.container import P3Container


def get_container(request: Request) -> P3Container:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="P3 service container is not initialized.",
        )
    return container
