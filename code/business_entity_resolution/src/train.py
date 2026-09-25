"""Train the entity matcher with a randomized LightGBM hyperparameter search.

Run from ``code/business_entity_resolution``::

    python src/train.py --data-dir ../../dataset --model-path models/lgbm_model.pkl
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocking import generate_candidates, evaluate_blocking_recall
from features import build_feature_matrix, get_feature_columns
from preprocessing import (
    load_source,
    load_ground_truth,
    parse_ground_truth,
    preprocess_dataframe,
)
from matcher import compute_entity_level_f05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BalancedNeuralMatcher(BaseEstimator, ClassifierMixin):
    """Small MLP trained with deterministic negative downsampling.

    ER candidate sets are highly imbalanced.  ``MLPClassifier`` did not support
    sample weights in the minimum supported scikit-learn version, so each fit
    keeps every positive and samples at most ``negative_ratio`` negatives per
    positive.  The validation-gated blend below corrects its probability scale.
    """

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        alpha=1e-3,
        negative_ratio=5,
        max_iter=250,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.negative_ratio = negative_ratio
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise ValueError("Neural matcher needs both positive and negative pairs")
        rng = np.random.RandomState(self.random_state)
        max_negatives = min(len(neg_idx), len(pos_idx) * self.negative_ratio)
        chosen_negatives = rng.choice(neg_idx, size=max_negatives, replace=False)
        sample_idx = np.r_[pos_idx, chosen_negatives]
        rng.shuffle(sample_idx)
        self.model_ = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=self.alpha,
                batch_size="auto",
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )),
        ])
        self.model_.fit(X[sample_idx], y[sample_idx])
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X))


class ValidationGatedEnsemble:
    """Serializable weighted probability ensemble for inference."""

    def __init__(self, lightgbm_model, extra_trees_model, neural_model, weights):
        self.lightgbm_model = lightgbm_model
        self.extra_trees_model = extra_trees_model
        self.neural_model = neural_model
        self.weights = weights

    def predict(self, X):
        X = np.asarray(X)
        return (
            self.weights["lightgbm"] * self.lightgbm_model.predict_proba(X)[:, 1]
            + self.weights["extra_trees"] * self.extra_trees_model.predict_proba(X)[:, 1]
            + self.weights["neural"] * self.neural_model.predict_proba(X)[:, 1]
        )


def train_randomized_model(
    features: pd.DataFrame,
    model_path=None,
    n_iter: int = 20,
    n_splits: int = 5,
    seed: int = 42,
    all_s1_ids=None,
    ground_truth=None,
):
    """Fit a grouped matcher and calibrate its threshold on OOF predictions.

    All candidate pairs for a Source 1 entity are kept in the same fold. This
    matches the macro entity-level leaderboard metric and prevents threshold
    selection from benefiting from in-sample probabilities.
    """
    if features.empty or features["label"].nunique() != 2:
        raise ValueError("Candidate features must contain both positive and negative labels")

    feature_cols = get_feature_columns(features)
    X = features[feature_cols]
    y = features["label"].astype(int)
    groups = features["s1_id"]
    n_groups = groups.nunique()
    if n_splits > n_groups:
        raise ValueError(f"n_splits={n_splits} exceeds the {n_groups} distinct source1 entities")

    estimator = LGBMClassifier(
        objective="binary", class_weight="balanced", n_jobs=1,
        random_state=seed, verbosity=-1, subsample_freq=1,
    )
    parameters = {
        "n_estimators": randint(200, 1201),
        "num_leaves": randint(15, 128),
        "max_depth": [-1, 5, 8, 12, 16],
        "learning_rate": loguniform(0.01, 0.15),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_samples": randint(10, 101),
        "reg_alpha": loguniform(1e-4, 10),
        "reg_lambda": loguniform(1e-4, 10),
    }
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=estimator, param_distributions=parameters, n_iter=n_iter,
        scoring="average_precision", cv=cv,
        refit=True, n_jobs=-1, random_state=seed, verbose=1, error_score="raise",
    )
    logger.info("Randomized search: %d settings, %d grouped folds, %d candidate pairs",
                n_iter, n_splits, len(features))
    search.fit(X, y, groups=groups)
    logger.info("Best mean grouped CV average precision: %.4f", search.best_score_)
    logger.info("Best parameters: %s", search.best_params_)

    # Each component is assessed only on records it did not train on.  This
    # makes it safe to retain a component only when it helps the real metric.
    lgbm_oof = cross_val_predict(
        search.best_estimator_, X, y, groups=groups, cv=cv,
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    extra_trees = ExtraTreesClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    neural = BalancedNeuralMatcher(random_state=seed)
    extra_oof = cross_val_predict(
        extra_trees, X, y, groups=groups, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    neural_oof = cross_val_predict(
        neural, X, y, groups=groups, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    weights, oof_probs, threshold, oof_f05 = find_best_ensemble(
        features,
        {"lightgbm": lgbm_oof, "extra_trees": extra_oof, "neural": neural_oof},
        all_s1_ids=all_s1_ids,
        ground_truth=ground_truth,
    )
    logger.info(
        "Best OOF macro entity F0.5: %.4f at threshold %.3f; ensemble weights=%s",
        oof_f05, threshold, weights,
    )

    # ``search.best_estimator_`` has already been refit on all pairs.
    extra_trees.fit(X, y)
    neural.fit(X, y)
    model = ValidationGatedEnsemble(search.best_estimator_, extra_trees, neural, weights)
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as model_file:
            pickle.dump({"model": model, "threshold": threshold}, model_file)
        logger.info("Saved compatible model artifact to %s", model_path)
    return model, threshold


def find_entity_threshold(features, probabilities, all_s1_ids=None, ground_truth=None):
    """Maximize macro entity-level F0.5 over an OOF probability threshold."""
    if all_s1_ids is None:
        all_s1_ids = features["s1_id"].unique().tolist()
    if ground_truth is None:
        ground_truth = {s1_id: set() for s1_id in all_s1_ids}
        for row in features.loc[features["label"] == 1, ["s1_id", "s2s3_id"]].itertuples(index=False):
            ground_truth.setdefault(row.s1_id, set()).add(row.s2s3_id)

    pairs = list(features[["s1_id", "s2s3_id"]].itertuples(index=False))
    thresholds = np.unique(np.r_[np.arange(0.05, 0.951, 0.01), 1.0])
    best_threshold, best_score = 1.0, -1.0
    for threshold in thresholds:
        predictions = {s1_id: set() for s1_id in all_s1_ids}
        for (s1_id, candidate_id), probability in zip(pairs, probabilities):
            if probability >= threshold:
                predictions.setdefault(s1_id, set()).add(candidate_id)
        score = compute_entity_level_f05(predictions, ground_truth)
        # Prefer a stricter cutoff for ties: false merges cost more in F0.5.
        if score >= best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def find_best_ensemble(features, component_probabilities, all_s1_ids=None, ground_truth=None):
    """Choose blend weights and threshold by grouped OOF macro F0.5.

    LightGBM starts as the baseline.  A secondary model receives non-zero weight
    only when it improves that baseline, avoiding the usual ensemble failure
    mode of diluting a strong tabular model with a weaker neural network.
    """
    baseline = component_probabilities["lightgbm"]
    baseline_threshold, baseline_score = find_entity_threshold(
        features, baseline, all_s1_ids=all_s1_ids, ground_truth=ground_truth
    )
    best_weights = {"lightgbm": 1.0, "extra_trees": 0.0, "neural": 0.0}
    best_probs = baseline
    best_threshold, best_score = baseline_threshold, baseline_score

    # Coarse weights reduce OOF overfitting while still allowing each alternate
    # model to be discarded entirely.  The final cutoff is optimized per blend.
    for lgbm_weight in np.arange(0.5, 1.01, 0.1):
        remaining = 1.0 - lgbm_weight
        for extra_weight in np.arange(0.0, remaining + 1e-9, 0.1):
            neural_weight = max(0.0, remaining - extra_weight)
            probs = (
                lgbm_weight * component_probabilities["lightgbm"]
                + extra_weight * component_probabilities["extra_trees"]
                + neural_weight * component_probabilities["neural"]
            )
            threshold, score = find_entity_threshold(
                features, probs, all_s1_ids=all_s1_ids, ground_truth=ground_truth
            )
            if score > best_score + 1e-12:
                best_weights = {
                    "lightgbm": round(float(lgbm_weight), 2),
                    "extra_trees": round(float(extra_weight), 2),
                    "neural": round(float(neural_weight), 2),
                }
                best_probs, best_threshold, best_score = probs, threshold, score
    return best_weights, best_probs, best_threshold, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Train the entity matcher using RandomizedSearchCV"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("../../dataset"))
    parser.add_argument("--model-path", type=Path, default=Path("models/lgbm_model.pkl"))
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Number of parameter settings sampled")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of grouped cross-validation folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-word", type=int, default=50)
    parser.add_argument("--top-k-char", type=int, default=30)
    parser.add_argument("--top-k-addr-char", type=int, default=20)
    parser.add_argument("--min-shared-tokens", type=int, default=2)
    args = parser.parse_args()

    if args.n_iter < 1 or args.cv < 2:
        parser.error("--n-iter must be positive and --cv must be at least 2")

    data_dir = args.data_dir
    logger.info("Loading and preprocessing training data from %s", data_dir)
    s1 = preprocess_dataframe(load_source(data_dir / "train" / "train_source1.tsv"))
    s2 = preprocess_dataframe(load_source(data_dir / "train" / "train_source2.tsv"))
    s3 = preprocess_dataframe(load_source(data_dir / "train" / "train_source3.tsv"))
    ground_truth = parse_ground_truth(
        load_ground_truth(data_dir / "train" / "train_ground_truth.tsv")
    )

    logger.info("Generating candidate pairs")
    candidates = generate_candidates(
        s1, s2, s3,
        top_k_word=args.top_k_word,
        top_k_char=args.top_k_char,
        top_k_addr_char=args.top_k_addr_char,
        min_shared_tokens=args.min_shared_tokens,
    )
    recall = evaluate_blocking_recall(candidates, ground_truth)
    logger.info("Blocking recall: %.4f", recall)

    features = build_feature_matrix(
        s1, pd.concat([s2, s3], ignore_index=True), candidates, ground_truth
    )
    if features.empty or features["label"].nunique() != 2:
        raise ValueError("Candidate features must contain both positive and negative labels")

    train_randomized_model(
        features, model_path=args.model_path, n_iter=args.n_iter,
        n_splits=args.cv, seed=args.seed,
        all_s1_ids=s1["entity_id"].tolist(), ground_truth=ground_truth,
    )


if __name__ == "__main__":
    main()
