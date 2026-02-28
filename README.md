# OpenCode Proxy

OpenAI-compatible API proxy for 10 LLM providers. One endpoint, one format, any backend.

## Quick Start

```bash
# Clone and install
git clone https://github.com/ScottRBK/opencode-proxy.git
cd opencode-proxy
uv sync

# Configure providers (only set keys for providers you want)
cp .env.example .env
# Edit .env with your API keys

# Run
uv run python main.py
# → http://localhost:4141
```

## Usage

Use it like any OpenAI-compatible API. Models are addressed as `provider/model-id`:

```bash
# Non-streaming
curl http://localhost:4141/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Streaming
curl http://localhost:4141/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Common model aliases work without the provider prefix: `gpt-4o`, `claude-sonnet-4-20250514`, `big-pickle`, etc.

### Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `GET /v1/models` | List available models |
| `GET /v1/models/{model_id}` | Get specific model info |
| `GET /health` | Provider availability |

### With OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4141/v1", api_key="unused")

response = client.chat.completions.create(
    model="groq/llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Supported Providers

| Provider | Model prefix | Env var |
|---|---|---|
| OpenAI | `openai/` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/` | `ANTHROPIC_API_KEY` |
| Groq | `groq/` | `GROQ_API_KEY` |
| Mistral | `mistral/` | `MISTRAL_API_KEY` |
| OpenRouter | `openrouter/` | `OPENROUTER_API_KEY` |
| Together AI | `togetherai/` | `TOGETHER_API_KEY` |
| DeepInfra | `deepinfra/` | `DEEPINFRA_API_KEY` |
| Perplexity | `perplexity/` | `PERPLEXITY_API_KEY` |
| xAI | `xai/` | `XAI_API_KEY` |
| OpenCode Zen | `opencode/` | `OPENCODE_API_KEY` |

Only providers with API keys configured will be active. Anthropic requests are automatically transformed between OpenAI and Anthropic message formats. All other providers use native OpenAI-compatible passthrough.

## Configuration

Server settings use the `OPENCODE_PROXY_` prefix:

```env
OPENCODE_PROXY_HOST=0.0.0.0  # default
OPENCODE_PROXY_PORT=4141      # default
OPENCODE_PROXY_LOG_LEVEL=INFO # default
```

## Tests

```bash
uv sync --group dev

# Unit + integration tests
uv run pytest tests/unit/ tests/integration/

# E2E tests (requires OPENCODE_API_KEY in .env)
uv run pytest -m e2e
```
