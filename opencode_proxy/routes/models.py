"""Models endpoint - list available models."""

import logging

from fastapi import APIRouter, HTTPException, Request

from opencode_proxy.models.openai_models import Model, ModelList

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Models"])


@router.get("/models")
async def list_models(request: Request) -> ModelList:
    """List available models across all configured providers."""
    registry = request.app.state.registry
    models = [
        Model(id=model_id, owned_by=provider_id)
        for model_id, provider_id in registry.list_models()
    ]
    return ModelList(data=models)


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, request: Request) -> Model:
    """Get a specific model by ID."""
    registry = request.app.state.registry
    try:
        provider_id, bare_model = registry.resolve_model(model_id)
        return Model(id=model_id, owned_by=provider_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
