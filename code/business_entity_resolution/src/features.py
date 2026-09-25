"""
Feature engineering for entity resolution.

For each candidate pair (S1 entity, S2/S3 entity), compute a rich set
of string similarity and structural features.
"""

import numpy as np
import pandas as pd
from collections import Counter
import logging
from rapidfuzz.distance import JaroWinkler, Levenshtein

logger = logging.getLogger(__name__)


# ── String similarity functions ──────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (0-1)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    # RapidFuzz provides the same edit-distance feature in optimized native
    # code. This matters when scoring tens of millions of blocked pairs.
    return float(Levenshtein.normalized_similarity(s1, s2))


def jaro_similarity(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions / 2) / matches) / 3
    return jaro


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return float(JaroWinkler.similarity(s1, s2, prefix_weight=p))


def jaccard_similarity(s1: str, s2: str) -> float:
    """Token-level Jaccard similarity."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 and not set2:
        return 1.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0


def token_overlap_ratio(s1: str, s2: str) -> float:
    """Fraction of tokens in s1 that appear in s2."""
    if not s1:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1)


def containment_similarity(s1: str, s2: str) -> float:
    """Max of both directional containment ratios."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    inter = len(tokens1 & tokens2)
    return max(inter / len(tokens1), inter / len(tokens2))


def cosine_similarity_tokens(s1: str, s2: str) -> float:
    """Token-level cosine similarity using term frequency vectors."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    counter1 = Counter(s1.split())
    counter2 = Counter(s2.split())
    all_tokens = set(counter1.keys()) | set(counter2.keys())
    dot = sum(counter1.get(t, 0) * counter2.get(t, 0) for t in all_tokens)
    mag1 = sum(v ** 2 for v in counter1.values()) ** 0.5
    mag2 = sum(v ** 2 for v in counter2.values()) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    ngrams1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    ngrams2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    inter = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(inter) / len(union)


def _extract_numbers(text: str) -> set:
    """Extract all numeric tokens from text."""
    import re
    return set(re.findall(r'\d+', text))


def number_overlap(s1: str, s2: str) -> float:
    """Fraction of numbers in common between two address strings."""
    nums1 = _extract_numbers(s1)
    nums2 = _extract_numbers(s2)
    if not nums1 and not nums2:
        return 1.0
    if not nums1 or not nums2:
        return 0.0
    inter = nums1 & nums2
    union = nums1 | nums2
    return len(inter) / len(union)


def sorted_token_similarity(s1: str, s2: str) -> float:
    """Sort tokens alphabetically and compare — handles word reordering."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    sorted1 = " ".join(sorted(s1.split()))
    sorted2 = " ".join(sorted(s2.split()))
    return jaro_winkler_similarity(sorted1, sorted2)


# ── Feature extraction ───────────────────────────────────────────────

def compute_pair_features(row_s1: pd.Series, row_s2s3: pd.Series) -> dict:
    """
    Compute all features for a single (S1, S2/S3) candidate pair.
    """
    name1 = row_s1.get("name_norm", "")
    name2 = row_s2s3.get("name_norm", "")
    addr1 = row_s1.get("addr_norm", "")
    addr2 = row_s2s3.get("addr_norm", "")
    comb1 = row_s1.get("combined_text", "")
    comb2 = row_s2s3.get("combined_text", "")

    features = {}

    # ── Name features ──
    features["name_levenshtein"] = levenshtein_similarity(name1, name2)
    features["name_jaro_winkler"] = jaro_winkler_similarity(name1, name2)
    features["name_jaccard"] = jaccard_similarity(name1, name2)
    features["name_token_overlap"] = token_overlap_ratio(name1, name2)
    features["name_containment"] = containment_similarity(name1, name2)
    features["name_cosine"] = cosine_similarity_tokens(name1, name2)
    features["name_char3gram"] = char_ngram_jaccard(name1, name2, 3)
    features["name_char4gram"] = char_ngram_jaccard(name1, name2, 4)
    features["name_sorted_jw"] = sorted_token_similarity(name1, name2)
    features["name_len_ratio"] = (
        min(len(name1), len(name2)) / max(len(name1), len(name2))
        if max(len(name1), len(name2)) > 0 else 1.0
    )
    features["name_token_count_diff"] = abs(len(name1.split()) - len(name2.split()))

    # ── Address features ──
    features["addr_levenshtein"] = levenshtein_similarity(addr1, addr2)
    features["addr_jaro_winkler"] = jaro_winkler_similarity(addr1, addr2)
    features["addr_jaccard"] = jaccard_similarity(addr1, addr2)
    features["addr_token_overlap"] = token_overlap_ratio(addr1, addr2)
    features["addr_containment"] = containment_similarity(addr1, addr2)
    features["addr_cosine"] = cosine_similarity_tokens(addr1, addr2)
    features["addr_char3gram"] = char_ngram_jaccard(addr1, addr2, 3)
    features["addr_number_overlap"] = number_overlap(addr1, addr2)
    nums1, nums2 = _extract_numbers(addr1), _extract_numbers(addr2)
    features["addr_has_conflicting_numbers"] = float(bool(nums1 and nums2 and not (nums1 & nums2)))
    features["addr_sorted_jw"] = sorted_token_similarity(addr1, addr2)
    features["addr_len_ratio"] = (
        min(len(addr1), len(addr2)) / max(len(addr1), len(addr2))
        if max(len(addr1), len(addr2)) > 0 else 1.0
    )

    # ── Combined features ──
    features["comb_jaccard"] = jaccard_similarity(comb1, comb2)
    features["comb_cosine"] = cosine_similarity_tokens(comb1, comb2)
    features["comb_char3gram"] = char_ngram_jaccard(comb1, comb2, 3)

    # ── Structural features ──
    features["both_empty_addr"] = 1.0 if (not addr1.strip() and not addr2.strip()) else 0.0
    features["one_empty_addr"] = 1.0 if (bool(addr1.strip()) != bool(addr2.strip())) else 0.0
    features["name_exact_match"] = 1.0 if name1 == name2 and name1 else 0.0
    features["candidate_is_source2"] = float(str(row_s2s3.get("entity_id", "")).startswith("S2-"))

    return features


def build_feature_matrix(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    candidates: dict,
    ground_truth: dict = None,
) -> pd.DataFrame:
    """
    Build a feature matrix for all candidate pairs.

    Args:
        s1_df: Preprocessed Source 1 dataframe
        s2s3_df: Preprocessed Source 2+3 dataframe (concatenated)
        candidates: dict s1_id -> set of candidate s2s3 ids
        ground_truth: optional dict s1_id -> set of true match ids (for labels)

    Returns:
        DataFrame with features, s1_id, s2s3_id, and optionally 'label'
    """
    logger.info("Building feature matrix...")

    # Index S2/S3 by entity_id for fast lookup
    s2s3_idx = s2s3_df.set_index("entity_id")
    s1_idx = s1_df.set_index("entity_id")

    rows = []
    total_pairs = sum(len(v) for v in candidates.values())
    processed = 0

    for s1_id, cand_ids in candidates.items():
        if s1_id not in s1_idx.index:
            continue
        row_s1 = s1_idx.loc[s1_id]

        for cand_id in cand_ids:
            if cand_id not in s2s3_idx.index:
                continue
            row_s2s3 = s2s3_idx.loc[cand_id]

            feats = compute_pair_features(row_s1, row_s2s3)
            feats["s1_id"] = s1_id
            feats["s2s3_id"] = cand_id

            if ground_truth is not None:
                true_matches = ground_truth.get(s1_id, set())
                feats["label"] = 1 if cand_id in true_matches else 0

            rows.append(feats)

            processed += 1
            if processed % 50000 == 0:
                logger.info(f"  Processed {processed}/{total_pairs} pairs...")

    df = pd.DataFrame(rows)
    logger.info(f"Feature matrix: {len(df)} pairs, {len(df.columns)} columns")

    if ground_truth is not None and "label" in df.columns:
        n_pos = (df["label"] == 1).sum()
        n_neg = (df["label"] == 0).sum()
        logger.info(f"  Positives: {n_pos}, Negatives: {n_neg}, Ratio: 1:{n_neg/max(n_pos,1):.1f}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get the list of feature columns (exclude IDs and label)."""
    exclude = {"s1_id", "s2s3_id", "label"}
    return [c for c in df.columns if c not in exclude]
