"""Estimators. Every model is wrapped so the same call signature works."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config


def make_model(name: str, cfg: Config):
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            class_weight=cfg.class_weight,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=1.0,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight=cfg.class_weight,
                        max_iter=5000,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    if name == "majority":
        return DummyClassifier(strategy="prior")
    raise ValueError(f"unknown model {name!r}")


def fit_predict(model, X_tr, y_tr, X_te, feature_names: list[str] | None = None) -> np.ndarray:
    """Fit and return P(allosteric) on the test block.

    Matrices are handed over as DataFrames when names are known: LightGBM
    records feature names at fit time, and predicting on a bare array then
    trips scikit-learn's name-consistency check.
    """
    if not isinstance(model, Pipeline):
        # The physchem block can carry NaN for molecules RDKit cannot parse.
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
        if feature_names is not None:
            X_tr = pd.DataFrame(X_tr, columns=feature_names)
            X_te = pd.DataFrame(X_te, columns=feature_names)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]
