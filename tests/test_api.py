
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def valid_features():
    return {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction():
    response = client.post(
        "/predict",
        json={"features": valid_features()},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert 0 <= body["confidence"] <= 1


def test_missing_feature():
    features = valid_features()
    del features["age"]

    response = client.post(
        "/predict",
        json={"features": features},
    )
    assert response.status_code == 422


def test_unexpected_feature():
    features = valid_features()
    features["unknown_feature"] = 1

    response = client.post(
        "/predict",
        json={"features": features},
    )
    assert response.status_code == 422


def test_null_feature():
    features = valid_features()
    features["ca"] = None

    response = client.post(
        "/predict",
        json={"features": features},
    )
    assert response.status_code == 200

def test_prediction_response_structure():
    response = client.post(
        "/predict",
        json={"features": valid_features()},
    )

    assert response.status_code == 200
    body = response.json()

    assert set(body.keys()) == {
        "prediction",
        "label",
        "confidence",
        "disclaimer",
    }
    assert body["prediction"] in [0, 1]
    assert isinstance(body["label"], str)
    assert isinstance(body["confidence"], (int, float))
    assert 0 <= body["confidence"] <= 1
    assert "not a medical diagnosis" in body["disclaimer"]


def test_non_finite_feature_is_rejected():
    features = valid_features()
    features["age"] = "Infinity"

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422


def test_boolean_feature_is_rejected():
    features = valid_features()
    features["age"] = True

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422


def test_invalid_request_body_is_rejected():
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422
