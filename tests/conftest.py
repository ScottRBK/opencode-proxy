"""Shared test fixtures."""

import pytest

from opencode_proxy.providers.registry import ProviderRegistry


@pytest.fixture
def registry():
    """Create a fresh ProviderRegistry."""
    return ProviderRegistry()
