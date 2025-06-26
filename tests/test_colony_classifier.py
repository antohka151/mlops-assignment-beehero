import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.models.colony_classifier import ColonyStrengthClassifier, ColonyStrengthClassifierConfig

@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    target = pd.Series(y, name="target")
    return df, target

@pytest.fixture
def default_config():
    return ColonyStrengthClassifierConfig(
        model_class_path="sklearn.ensemble.RandomForestClassifier",
        hyperparameters={"n_estimators": 10, "random_state": 42}
    )

def test_init(default_config):
    clf = ColonyStrengthClassifier(default_config)
    assert clf.config == default_config
    assert clf._model is None

def test_fit_sets_model_and_classes(sample_data, default_config):
    X, y = sample_data
    clf = ColonyStrengthClassifier(default_config)
    clf.fit(X, y)
    assert clf._model is not None
    assert hasattr(clf, "classes_")
    assert set(clf.classes_) == set(np.unique(y))

def test_predict_after_fit(sample_data, default_config):
    X, y = sample_data
    clf = ColonyStrengthClassifier(default_config)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset(set(clf.classes_))

def test_predict_before_fit_raises(sample_data, default_config):
    X, _ = sample_data
    clf = ColonyStrengthClassifier(default_config)
    with pytest.raises(Exception):
        clf.predict(X)

def test_feature_importances_property(sample_data, default_config):
    X, y = sample_data
    clf = ColonyStrengthClassifier(default_config)
    clf.fit(X, y)
    importances = clf.feature_importances_
    assert isinstance(importances, np.ndarray)
    assert len(importances) == X.shape[1]

def test_feature_importances_raises_for_model_without_importances(default_config, sample_data):
    # Use a model that does not have feature_importances_
    config = ColonyStrengthClassifierConfig(
        model_class_path="sklearn.svm.SVC",
        hyperparameters={}
    )
    X, y = sample_data
    clf = ColonyStrengthClassifier(config)
    clf.fit(X, y)
    with pytest.raises(AttributeError):
        _ = clf.feature_importances_
