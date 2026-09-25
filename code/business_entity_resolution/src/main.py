"""
Main pipeline for the Business Entity Resolution Challenge.

Usage:
    python src/main.py --data-dir ../../dataset --output-dir ../../output

Steps:
    1. Load & preprocess all source files
    2. (Training) Generate candidates, build features, train model
    3. (Test) Generate candidates, build features, predict matches
    4. Write output files: matching_results.tsv, candidate_pairs.tsv
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    load_source, load_ground_truth,
    preprocess_dataframe, parse_ground_truth,
)
from blocking import generate_candidates, evaluate_blocking_recall
from features import build_feature_matrix, get_feature_columns
from matcher import (
    predict_matches, load_model,
    compute_entity_level_f05,
)
from train import train_randomized_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_matching_results(predictions: dict, output_path: str):
    """Write matching_results.tsv."""
    rows = []
    for s1_id in sorted(predictions.keys()):
        matched = predictions[s1_id]
        matched_str = ",".join(sorted(matched)) if matched else ""
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote matching results to {output_path} ({len(df)} rows)")


def write_candidate_pairs(candidates: dict, output_path: str):
    """Write candidate_pairs.tsv."""
    rows = []
    for s1_id in sorted(candidates.keys()):
        cands = candidates[s1_id]
        cands_str = ",".join(sorted(cands)) if cands else ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cands_str})
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote candidate pairs to {output_path} ({len(df)} rows)")


def run_validation_split(
    s1_df, s2_df, s3_df, ground_truth, val_fraction=0.2, seed=42
):
    """
    Hold out a fraction of S1 entities for validation.
    Returns (train_gt, val_gt, train_s1_ids, val_s1_ids).
    """
    np.random.seed(seed)
    s1_ids = list(ground_truth.keys())
    np.random.shuffle(s1_ids)
    split = int(len(s1_ids) * (1 - val_fraction))
    train_ids = set(s1_ids[:split])
    val_ids = set(s1_ids[split:])

    train_gt = {k: v for k, v in ground_truth.items() if k in train_ids}
    val_gt = {k: v for k, v in ground_truth.items() if k in val_ids}

    return train_gt, val_gt, train_ids, val_ids


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", type=str, default="../../dataset",
                        help="Path to dataset/ directory")
    parser.add_argument("--output-dir", type=str, default="../../output",
                        help="Path to output/ directory")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Path to save/load models")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use saved model)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation on training data")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of training data for validation")
    parser.add_argument("--top-k-word", type=int, default=50,
                        help="Top-K for word TF-IDF blocking")
    parser.add_argument("--top-k-char", type=int, default=30,
                        help="Top-K for char n-gram TF-IDF blocking")
    parser.add_argument("--top-k-addr-char", type=int, default=20,
                        help="Top-K for address char n-gram TF-IDF blocking")
    parser.add_argument("--min-shared-tokens", type=int, default=2,
                        help="Min shared tokens for token blocking")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override prediction threshold (default: auto from CV)")
    parser.add_argument("--search-iterations", type=int, default=20,
                        help="Randomized hyperparameter settings to evaluate")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Grouped cross-validation folds for model search")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading data...")

    train_s1 = load_source(data_dir / "train" / "train_source1.tsv")
    train_s2 = load_source(data_dir / "train" / "train_source2.tsv")
    train_s3 = load_source(data_dir / "train" / "train_source3.tsv")
    train_gt_df = load_ground_truth(data_dir / "train" / "train_ground_truth.tsv")
    ground_truth = parse_ground_truth(train_gt_df)

    logger.info(f"Train: S1={len(train_s1)}, S2={len(train_s2)}, S3={len(train_s3)}, GT={len(ground_truth)}")

    # ── 2. Preprocess ────────────────────────────────────────────────
    logger.info("Preprocessing...")
    train_s1 = preprocess_dataframe(train_s1)
    train_s2 = preprocess_dataframe(train_s2)
    train_s3 = preprocess_dataframe(train_s3)

    # ── 3. Train or Load Model ───────────────────────────────────────
    model_path = model_dir / "lgbm_model.pkl"

    if args.validate_only:
        logger.info("=" * 60)
        logger.info("Running validation split...")
        train_gt, val_gt, train_s1_ids, val_s1_ids = run_validation_split(
            train_s1, train_s2, train_s3, ground_truth,
            val_fraction=args.val_fraction,
        )

        # Train on train split
        train_s1_split = train_s1[train_s1["entity_id"].isin(train_s1_ids)]
        val_s1_split = train_s1[train_s1["entity_id"].isin(val_s1_ids)]

        # Generate candidates for train split
        logger.info("Generating train candidates...")
        train_candidates = generate_candidates(
            train_s1_split, train_s2, train_s3,
            top_k_word=args.top_k_word,
            top_k_char=args.top_k_char,
            top_k_addr_char=args.top_k_addr_char,
            min_shared_tokens=args.min_shared_tokens,
        )
        train_recall = evaluate_blocking_recall(train_candidates, train_gt)

        # Build features for train split
        s2s3_combined = pd.concat([train_s2, train_s3], ignore_index=True)
        train_features = build_feature_matrix(
            train_s1_split, s2s3_combined, train_candidates, train_gt
        )
        feature_cols = get_feature_columns(train_features)

        # Train model
        model, threshold = train_randomized_model(
            train_features, n_iter=args.search_iterations, n_splits=args.cv_folds,
            all_s1_ids=list(train_s1_ids), ground_truth=train_gt,
        )
        if args.threshold is not None:
            threshold = args.threshold

        # Generate candidates for val split
        logger.info("Generating validation candidates...")
        val_candidates = generate_candidates(
            val_s1_split, train_s2, train_s3,
            top_k_word=args.top_k_word,
            top_k_char=args.top_k_char,
            top_k_addr_char=args.top_k_addr_char,
            min_shared_tokens=args.min_shared_tokens,
        )
        val_recall = evaluate_blocking_recall(val_candidates, val_gt)

        # Build features for val split
        val_features = build_feature_matrix(
            val_s1_split, s2s3_combined, val_candidates
        )

        # Predict
        val_predictions = predict_matches(
            model, val_features, feature_cols, threshold,
            list(val_s1_ids),
        )

        # Evaluate
        val_f05 = compute_entity_level_f05(val_predictions, val_gt)
        logger.info("=" * 60)
        logger.info(f"VALIDATION RESULTS:")
        logger.info(f"  Blocking recall:  {val_recall:.4f}")
        logger.info(f"  Entity-level F0.5: {val_f05:.4f}")
        logger.info("=" * 60)
        return

    if args.skip_train:
        logger.info(f"Loading saved model from {model_path}")
        model, threshold = load_model(str(model_path))
    else:
        logger.info("=" * 60)
        logger.info("Training pipeline...")

        # Generate candidates on full training data
        train_candidates = generate_candidates(
            train_s1, train_s2, train_s3,
            top_k_word=args.top_k_word,
            top_k_char=args.top_k_char,
            top_k_addr_char=args.top_k_addr_char,
            min_shared_tokens=args.min_shared_tokens,
        )
        blocking_recall = evaluate_blocking_recall(train_candidates, ground_truth)

        # Build features
        s2s3_combined = pd.concat([train_s2, train_s3], ignore_index=True)
        train_features = build_feature_matrix(
            train_s1, s2s3_combined, train_candidates, ground_truth
        )
        feature_cols = get_feature_columns(train_features)

        # Train
        model, threshold = train_randomized_model(
            train_features,
            model_path=model_path,
            n_iter=args.search_iterations,
            n_splits=args.cv_folds,
            all_s1_ids=train_s1["entity_id"].tolist(),
            ground_truth=ground_truth,
        )

    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"Using override threshold: {threshold}")

    # ── 4. Test prediction ───────────────────────────────────────────
    test_s1_path = data_dir / "test" / "test_source1.tsv"
    if not test_s1_path.exists():
        logger.info("No test data found. Skipping test prediction.")
        return

    logger.info("=" * 60)
    logger.info("Running test prediction...")

    test_s1 = load_source(data_dir / "test" / "test_source1.tsv")
    test_s2 = load_source(data_dir / "test" / "test_source2.tsv")
    test_s3 = load_source(data_dir / "test" / "test_source3.tsv")

    logger.info(f"Test: S1={len(test_s1)}, S2={len(test_s2)}, S3={len(test_s3)}")

    test_s1 = preprocess_dataframe(test_s1)
    test_s2 = preprocess_dataframe(test_s2)
    test_s3 = preprocess_dataframe(test_s3)

    # Generate test candidates
    test_candidates = generate_candidates(
        test_s1, test_s2, test_s3,
        top_k_word=args.top_k_word,
        top_k_char=args.top_k_char,
        min_shared_tokens=args.min_shared_tokens,
    )

    # Build test features
    test_s2s3 = pd.concat([test_s2, test_s3], ignore_index=True)
    # Reload feature cols from training
    if not args.skip_train:
        pass  # feature_cols already set
    else:
        # Need to infer feature cols — use same set
        feature_cols = get_feature_columns(
            build_feature_matrix(
                train_s1.head(1),
                pd.concat([train_s2.head(1), train_s3.head(1)]),
                {train_s1.iloc[0]["entity_id"]: set()},
            )
        )

    test_features = build_feature_matrix(
        test_s1, test_s2s3, test_candidates
    )

    # Predict
    test_predictions = predict_matches(
        model, test_features, feature_cols, threshold,
        test_s1["entity_id"].tolist(),
    )

    # ── 5. Write outputs ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Writing output files...")

    write_matching_results(test_predictions, output_dir / "matching_results.tsv")
    write_candidate_pairs(test_candidates, output_dir / "candidate_pairs.tsv")

    logger.info("=" * 60)
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
