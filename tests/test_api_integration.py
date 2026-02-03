import pytest
from fastapi.testclient import TestClient
from neurobridge.api import app
import asyncio
import time

def test_websocket_signals_broadcast():
    # Use context manager to trigger startup/shutdown events
    with TestClient(app) as client:
        with client.websocket_connect("/ws/signals") as websocket:
            # Wait for first frame
            data = websocket.receive_json()
            
            # Check for Holographic Frame keys
            assert "timestamp" in data
            assert "latency_ms" in data
            assert "signal_snapshot" in data
            assert "health" in data
            
            # Verify signal shape (128 channels)
            assert len(data["signal_snapshot"]) == 128
            
            # Check health structure
            assert "bad_channels" in data["health"]

def test_get_status_with_engine():
    with TestClient(app) as client:
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["engine_running"] == True