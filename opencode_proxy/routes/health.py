"""Health and info endpoints."""

import logging

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health(request: Request) -> dict:
    """Health check with available provider info."""
    registry = request.app.state.registry
    return {
        "status": "ok",
        "service": "opencode-proxy",
        "available_providers": registry.available_providers(),
    }


@router.get("/")
async def root() -> dict:
    """Service info."""
    return {
        "service": "opencode-proxy",
        "version": "0.1.0",
        "docs": "/docs",
    }
