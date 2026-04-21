import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.api import app

client = TestClient(app)

def test_root_endpoint():
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health_endpoint():
    """Тест health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_missing_text_field():
    """Тест с отсутствующим полем text (должен вернуть 422)"""
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_predict_endpoint_exists():
    """Проверка, что эндпоинт /predict существует"""
    response = client.post("/predict", json={"text": "test"})
    # Может быть 200 (если модель загружена) или 503 (если нет)
    # Главное, что не 404
    assert response.status_code in [200, 503]

def test_response_structure():
    """Проверка структуры ответа при успешном запросе"""
    # Пропускаем если модель не загружена
    response = client.post("/predict", json={"text": "Hello test"})
    if response.status_code == 200:
        data = response.json()
        assert "is_spam" in data
        assert "label" in data
        assert "confidence" in data