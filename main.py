"""OpenCode Proxy - OpenAI-compatible proxy for multiple LLM providers."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from opencode_proxy.config import settings
from opencode_proxy.providers.registry import ProviderRegistry
from opencode_proxy.routes import chat_completions, health, models

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - init registry, log providers."""
    registry = ProviderRegistry()
    app.state.registry = registry

    available = registry.available_providers()
    if available:
        logger.info(f"Available providers: {', '.join(available)}")
    else:
        logger.warning("No provider API keys found. Set environment variables to enable providers.")

    print(f"\nOpenCode Proxy ready at http://localhost:{settings.PORT}")
    print(f"Available providers: {', '.join(available) if available else 'none'}")
    print(f"API docs: http://localhost:{settings.PORT}/docs\n")

    yield

    await registry.close_all()
    logger.info("Shutdown complete")


app = FastAPI(
    title="OpenCode Proxy",
    description="OpenAI-compatible proxy for multiple LLM providers",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_completions.router)
app.include_router(models.router)
app.include_router(health.router)


def main():
    """Run the proxy server."""
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
