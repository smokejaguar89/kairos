import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Fixture providing a test client for the FastAPI app."""
    return TestClient(app)
