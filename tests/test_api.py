from app.app import app


def test_home_route():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200


def test_predict_route_missing_text():
    client = app.test_client()
    response = client.post("/predict", json={})
    assert response.status_code == 400