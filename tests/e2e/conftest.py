"""E2E test fixtures - starts a real proxy server."""

import threading
import time

import httpx
import pytest
import uvicorn

from opencode_proxy.config import settings

# Skip all E2E tests if no OPENCODE_API_KEY
pytestmark = pytest.mark.skipif(
    not settings.OPENCODE_API_KEY,
    reason="OPENCODE_API_KEY not set",
)


class ServerThread(threading.Thread):
    """Run uvicorn in a background thread."""

    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.server = None

    def run(self):
        config = uvicorn.Config(
            "main:app",
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self.server = uvicorn.Server(config)
        self.server.run()

    def stop(self):
        if self.server:
            self.server.should_exit = True


@pytest.fixture(scope="session")
def server_url():
    """Start the proxy on a random port and return its base URL."""
    port = 14141  # Use a distinct port for tests
    host = "127.0.0.1"

    thread = ServerThread(host, port)
    thread.start()

    # Wait for server to be ready
    base_url = f"http://{host}:{port}"
    for _ in range(50):
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(0.1)
    else:
        pytest.fail("Server did not start in time")

    yield base_url

    thread.stop()


@pytest.fixture
def api_client(server_url):
    """Provide an httpx client pointed at the test server."""
    with httpx.Client(base_url=server_url, timeout=30.0) as client:
        yield client
