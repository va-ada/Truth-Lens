"""
TruthLens — Model Trainer
===========================
Trains a STACKING ENSEMBLE of SVM, Logistic Regression, and Random Forest.

Key improvements over basic approaches:
  - Stacking ensemble with LR meta-learner (better than majority voting)
  - GridSearchCV for hyperparameter tuning with stratified CV
  - class_weight='balanced' to handle class imbalance
  - Full model persistence via joblib
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TruthLensTrainer:
    """
    Trains individual models and a stacking ensemble.

    The stacking ensemble uses a Logistic Regression meta-learner
    that learns optimal weights for each base model's predictions,
    rather than naive majority voting.
    """

    def __init__(self):
        self.base_models = {}          # Fitted base models
        self.best_params = {}          # Best hyperparams from GridSearch
        self.ensemble = None           # Fitted stacking ensemble
        self.training_results = {}     # Performance during training

    def train_base_models(self, X_train, y_train):
        """
        Train individual base models with hyperparameter tuning.

        Args:
            X_train: Feature matrix (n_samples, n_features)
            y_train: Labels (n_samples,)

        Returns:
            dict of fitted base models
        """
        print("\n" + "=" * 60)
        print("TRAINING BASE MODELS")
        print("=" * 60)

        cv = StratifiedKFold(
            n_splits=config.CV_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_STATE,
        )

        models_config = {
            # LinearSVC: O(n) training — 100x faster than SVC(kernel='rbf') on 40k+ samples.
            # CalibratedClassifierCV wrapper adds predict_proba support (required for stacking).
            "SVM": (
                CalibratedClassifierCV(
                    LinearSVC(random_state=config.RANDOM_STATE),
                    cv=3,
                ),
                config.get_svm_params(),
            ),
            "LogisticRegression": (
                LogisticRegression(random_state=config.RANDOM_STATE),
                config.get_lr_params(),
            ),
            "RandomForest": (
                RandomForestClassifier(random_state=config.RANDOM_STATE),
                config.get_rf_params(),
            ),
        }

        for name, (model, param_grid) in models_config.items():
            print(f"\n--- Training {name} ---")
            start = time.time()

            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring="f1",           # Optimize for F1, not accuracy
                n_jobs=-1,
                verbose=0,
                refit=True,
            )
            grid.fit(X_train, y_train)

            elapsed = time.time() - start
            self.base_models[name] = grid.best_estimator_
            self.best_params[name] = grid.best_params_

            # Evaluate on training data (to compare with test later)
            y_pred = grid.best_estimator_.predict(X_train)
            train_f1 = f1_score(y_train, y_pred)
            train_acc = accuracy_score(y_train, y_pred)

            self.training_results[name] = {
                "best_cv_f1": grid.best_score_,
                "train_f1": train_f1,
                "train_accuracy": train_acc,
                "best_params": grid.best_params_,
                "training_time": elapsed,
            }

            print(f"  Best params: {grid.best_params_}")
            print(f"  CV F1: {grid.best_score_:.4f}")
            print(f"  Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Time: {elapsed:.1f}s")

        return self.base_models

    def train_ensemble(self, X_train, y_train):
        """
        Train stacking ensemble using base models.

        The stacking classifier:
        1. Each base model makes predictions using cross-validation
        2. A meta-learner (Logistic Regression) learns to combine predictions
        3. This is MUCH better than simple majority voting

        Args:
            X_train: Feature matrix
            y_train: Labels

        Returns:
            Fitted StackingClassifier
        """
        print(f"\n--- Training Stacking Ensemble ---")
        start = time.time()

        # Define estimators for stacking using best params from GridSearch.
        svm_best = self.best_params.get("SVM", {})
        best_svm_C = svm_best.get("estimator__C", 1)
        best_svm_cw = svm_best.get("estimator__class_weight", "balanced")
        best_svm_iter = svm_best.get("estimator__max_iter", 2000)

        estimators = [
            ("svm", CalibratedClassifierCV(
                LinearSVC(
                    C=best_svm_C,
                    class_weight=best_svm_cw,
                    max_iter=best_svm_iter,
                    random_state=config.RANDOM_STATE,
                ),
                cv=3,
            )),
            ("lr", LogisticRegression(
                **self.best_params.get("LogisticRegression", {"C": 1, "penalty": "l2", "max_iter": 2000, "solver": "lbfgs", "class_weight": "balanced"}),
                random_state=config.RANDOM_STATE,
            )),
            ("rf", RandomForestClassifier(
                **self.best_params.get("RandomForest", {"n_estimators": 200, "max_depth": 20, "class_weight": "balanced"}),
                random_state=config.RANDOM_STATE,
            )),
        ]

        self.ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=config.RANDOM_STATE,
            ),
            cv=config.CV_FOLDS,
            stack_method="predict_proba",     # Use probabilities, not raw predictions
            n_jobs=-1,
            passthrough=False,
        )

        self.ensemble.fit(X_train, y_train)
        elapsed = time.time() - start

        # Training performance
        y_pred = self.ensemble.predict(X_train)
        train_f1 = f1_score(y_train, y_pred)
        train_acc = accuracy_score(y_train, y_pred)

        self.training_results["Ensemble"] = {
            "train_f1": train_f1,
            "train_accuracy": train_acc,
            "training_time": elapsed,
        }

        print(f"  Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        return self.ensemble

    def train_debiased(self, X_train, y_train, bias_features_idx, strategy="zero"):
        """Re-fit the stacking ensemble with bias-correlated columns neutralised.

        Used by the A2 debiasing loop. SHAP+audit_bias have already identified
        which feature columns correlate with source-leakage. We do NOT drop the
        columns (would break the saved feature_engine.pkl shape) — instead we
        zero them in a copy of the training matrix and the inference path zeroes
        the same columns at predict time.

        Args:
            X_train: scaled feature matrix (n_samples, n_features)
            y_train: binary labels
            bias_features_idx: list of column indices to neutralise
            strategy: "zero" (default) zeros the columns. "downweight" scales them by 0.1.

        Returns the fitted ensemble. Also stashes `bias_features_idx` on the
        instance so callers can persist it alongside the model.
        """
        if not bias_features_idx:
            print("[DEBIAS] No bias features to neutralise — running standard training.")
            self.train_base_models(X_train, y_train)
            return self.train_ensemble(X_train, y_train)

        X_dbg = X_train.copy()
        if strategy == "zero":
            X_dbg[:, bias_features_idx] = 0.0
        elif strategy == "downweight":
            X_dbg[:, bias_features_idx] *= 0.1
        else:
            raise ValueError(f"Unknown debiasing strategy: {strategy}")

        print(f"[DEBIAS] Neutralised {len(bias_features_idx)} columns "
              f"with strategy='{strategy}'. Re-training base models + ensemble...")
        self.train_base_models(X_dbg, y_train)
        self.train_ensemble(X_dbg, y_train)
        # Stash the indices so save() / consumers can recover them.
        self.bias_features_idx = list(bias_features_idx)
        return self.ensemble

    def predict(self, X, model_name="Ensemble"):
        """Predict labels using specified model."""
        model = self._get_model(model_name)
        return model.predict(X)

    def predict_proba(self, X, model_name="Ensemble"):
        """Predict probabilities using specified model."""
        model = self._get_model(model_name)
        return model.predict_proba(X)

    def _get_model(self, name):
        """Get model by name."""
        if name == "Ensemble":
            if self.ensemble is None:
                raise RuntimeError("Ensemble not trained yet.")
            return self.ensemble
        elif name in self.base_models:
            return self.base_models[name]
        else:
            raise ValueError(f"Unknown model: {name}. Available: {list(self.base_models.keys()) + ['Ensemble']}")

    def evaluate_on_test(self, X_test, y_test):
        """
        Evaluate all models on test set.

        Returns a comparison DataFrame with metrics for each model.
        """
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)

        results = []
        all_models = list(self.base_models.keys())
        if self.ensemble is not None:
            all_models.append("Ensemble")

        for name in all_models:
            model = self._get_model(name)
            y_pred = model.predict(X_test)

            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = np.nan

            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "AUC-ROC": auc,
            }
            results.append(metrics)

            print(f"\n--- {name} ---")
            print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

        results_df = pd.DataFrame(results)
        print("\n=== COMPARISON TABLE ===")
        print(results_df.to_string(index=False, float_format="{:.4f}".format))

        return results_df

    def save(self, path=None):
        """Save all models to disk."""
        if path is None:
            path = config.MODELS_DIR

        os.makedirs(path, exist_ok=True)

        # Save base models
        for name, model in self.base_models.items():
            model_path = os.path.join(path, f"{name.lower()}.joblib")
            joblib.dump(model, model_path)
            print(f"[MODEL] Saved {name}: {model_path}")

        # Save ensemble
        if self.ensemble is not None:
            ens_path = os.path.join(path, "ensemble.joblib")
            joblib.dump(self.ensemble, ens_path)
            print(f"[MODEL] Saved Ensemble: {ens_path}")

        # Save best params and results
        meta_path = os.path.join(path, "training_meta.joblib")
        joblib.dump({
            "best_params": self.best_params,
            "training_results": self.training_results,
        }, meta_path)

    def load(self, path=None):
        """Load all models from disk."""
        if path is None:
            path = config.MODELS_DIR

        # Load base models
        for name in ["SVM", "LogisticRegression", "RandomForest"]:
            model_path = os.path.join(path, f"{name.lower()}.joblib")
            if os.path.exists(model_path):
                self.base_models[name] = joblib.load(model_path)
                print(f"[MODEL] Loaded {name}")

        # Load ensemble
        ens_path = os.path.join(path, "ensemble.joblib")
        if os.path.exists(ens_path):
            self.ensemble = joblib.load(ens_path)
            print("[MODEL] Loaded Ensemble")

        # Load meta
        meta_path = os.path.join(path, "training_meta.joblib")
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.best_params = meta.get("best_params", {})
            self.training_results = meta.get("training_results", {})

        return self


if __name__ == "__main__":
    # Quick test with synthetic data
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, n_features=50, random_state=42)
    trainer = TruthLensTrainer()
    trainer.train_base_models(X[:400], y[:400])
    trainer.train_ensemble(X[:400], y[:400])
    results = trainer.evaluate_on_test(X[400:], y[400:])
