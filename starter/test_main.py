import json

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_message():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "welcome to the API"}


def test_less_than_or_equal_50k():
    data = {
        "age": 56,
        "workclass": "Private",
        "fnlgt": 169133,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Other-service",
        "relationship": "Husband",
        "race": " White",
        "sex": "Male",
        "hours-per-week": "50",
        "native-country": "Yugoslavia",
    } 
    r = client.post("/inference/", json=data)
    assert r.status_code == 200
    assert r.json() == {"salary": "<=50k"}


def test_morethan_50k():
    data = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 368561,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": " White",
        "sex": "Male",
        "hours-per-week": "55",
        "native-country": " United-States",
    } 
    r = client.post("/inference/", json=data)
    assert r.status_code == 200
    assert r.json() == {"salary": ">50k"}
