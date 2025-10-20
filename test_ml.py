import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model,
)


# TODO: add necessary import
@pytest.fixture(scope="module")
def sample_data():
    data = pd.read_csv("data/census.csv").sample(100, random_state=42)
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    return data, cat_features


# TODO: implement the first test. Change the function name and input as needed
def test_one(sample_data):
    """
    Test that train_model returns a scikit-learn estimator of the correct type
    """
    data, cat_features = sample_data
    X, y, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression), (
        "Model is not a LogisticRegression instance."
    )


def test_two(sample_data):
    """
    Test that inference returns predictions of correct shape and type.
    """
    data, cat_features = sample_data
    X, y, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape, "Prediction shape mismatch."
    assert set(np.unique(preds)).issubset({0, 1}), (
        "Predictions must be binary (0 or 1)."
    )


# TODO: implement the second test. Change the function name and input as neede
def test_three(sample_data):
    """
    Test compute_model_metrics produces consistent and reasonable outputs.
    """
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    p, r, f = compute_model_metrics(y_true, y_pred)
    assert (p, r, f) == (1.0, 1.0, 1.0), (
        "Metrics should be perfect for identical predictions."
    )


# TODO: implement the third test. Change the function name and input as needed
def test_four(sample_data):
    """
    Test that process_data returns arrays with compatible shapes.
    """
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    assert X.shape[0] == y.shape[0], (
        "Mismatch in number of samples between X and y."
    )
    assert encoder is not None and lb is not None, (
        "Encoder or label binarizer not returned."
    )


def test_five(sample_data):
    """
    Test that a saved model can be loaded and produces identical predictions.
    """
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    model = train_model(X, y)

    preds_before = inference(model, X)
    project_path = Path(__file__).resolve().parent
    model_path = project_path / "model.pkl"
    save_model(model, model_path)
    loaded_model = load_model(model_path)

    preds_after = inference(loaded_model, X)

    assert np.array_equal(preds_before, preds_after), (
        "Loaded model predictions differ from original."
    )
