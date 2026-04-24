def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_hello_endpoint(client):
    """Test the hello endpoint."""
    response = client.get("/api/hello")
    assert response.status_code == 200
    assert "message" in response.json()
