import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_list_stations():
    """Test stations list endpoint"""
    response = client.get("/api/v1/stations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_list_crimes():
    """Test crimes list endpoint"""
    response = client.get("/api/v1/crimes?limit=10")
    assert response.status_code == 200
    assert isinstance(response.json(), list)



