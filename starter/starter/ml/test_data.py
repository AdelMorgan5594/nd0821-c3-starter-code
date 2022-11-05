import pytest
import pandas as pd
from .data import process_data



cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture
def data():
    df = pd.read_csv('./starter/data/census_clean.csv')
    return df

def test_process_data(data):
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    assert X.shape == (32561, 107)
    assert y.shape == (32561,)