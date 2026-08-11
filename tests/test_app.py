from fastapi.testclient import TestClient

from rag_config_errores import app


client = TestClient(app)


def test_suma():
    resultado = 2 + 3

    assert resultado == 5


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"state": "healthy"}