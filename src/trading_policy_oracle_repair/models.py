from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge


REGRESSION_MODEL_NAMES = ("ridge", "gradient_boosting")
CLASSIFICATION_MODEL_NAME = "logistic"


@dataclass(frozen=True)
class ModelFitInfo:
    model_name: str
    task: str
    train_rows: int
    fallback: str | None = None


def _as_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be 1D and aligned with X")
    if len(X) == 0:
        raise ValueError("empty training data")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite after preprocessing")
    return X, y


def fit_regressor(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    ridge_alpha: float = 1.0,
    gbt_learning_rate: float = 0.05,
    gbt_max_iter: int = 80,
    gbt_max_leaf_nodes: int = 15,
    gbt_min_samples_leaf: int = 50,
    gbt_l2_regularization: float = 0.001,
):
    X, y = _as_xy(X, y)
    if model_name == "ridge":
        model = Ridge(alpha=float(ridge_alpha))
    elif model_name == "gradient_boosting":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(gbt_learning_rate),
            max_iter=int(gbt_max_iter),
            max_leaf_nodes=int(gbt_max_leaf_nodes),
            min_samples_leaf=int(gbt_min_samples_leaf),
            l2_regularization=float(gbt_l2_regularization),
            early_stopping=False,
            random_state=int(seed),
        )
    else:
        raise ValueError(f"unknown regression model: {model_name}")
    model.fit(X, y)
    return model


def fit_high_regret_classifier(
    X: np.ndarray,
    y_binary: np.ndarray,
    *,
    seed: int,
    logistic_c: float = 1.0,
    logistic_max_iter: int = 300,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y_binary, dtype=int)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) == 0:
        raise ValueError("invalid classification training data")
    if not np.isfinite(X).all():
        raise ValueError("X must be finite")
    classes = np.unique(y)
    if not set(classes).issubset({0, 1}):
        raise ValueError("classification target must be binary")
    if len(classes) < 2:
        model = DummyClassifier(strategy="constant", constant=int(classes[0]))
        model.fit(X, y)
        return model, "single_class_dummy"
    model = LogisticRegression(
        C=float(logistic_c),
        max_iter=int(logistic_max_iter),
        solver="lbfgs",
        random_state=int(seed),
    )
    model.fit(X, y)
    return model, None


def predict_probability(model: Any, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        if proba.ndim != 2:
            raise ValueError("predict_proba returned invalid shape")
        if 1 in classes:
            return proba[:, int(np.where(classes == 1)[0][0])]
        return np.zeros(len(X), dtype=float)
    pred = np.asarray(model.predict(X), dtype=float)
    return np.clip(pred, 0.0, 1.0)
