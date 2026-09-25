# Business Entity Resolution

## Setup

```bash
pip install -r requirements.txt
```

For the supplied multi-million-row dataset, see [COLAB.md](COLAB.md) for the
high-RAM Colab workflow and recommended first training run.

## Directory Structure

Place the dataset at the expected path relative to this directory:

```
amazonml/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── main.py
│       │   ├── preprocessing.py
│       │   ├── blocking.py
│       │   ├── features.py
│       │   └── matcher.py
│       ├── requirements.txt
│       └── README.md
└── utils/
    └── validate_submission.py
```

## Usage

All commands run from `code/business_entity_resolution/`.

### Full pipeline (train + predict on test set)

```bash
python src/main.py --data-dir ../../dataset --output-dir ../../output
```

### Validation only (evaluate on held-out training split)

```bash
python src/main.py --data-dir ../../dataset --output-dir ../../output --validate-only --val-fraction 0.2
```

### Train with randomized hyperparameter search

```bash
python src/train.py --data-dir ../../dataset --model-path models/lgbm_model.pkl
```

`train.py` provides a standalone training command. The standard `main.py`
pipeline also uses the same randomized LightGBM search during training.
Cross-validation keeps all candidate pairs for each source-1 entity in the same
fold. Tune the standard pipeline's search with `--search-iterations` and
`--cv-folds`; the standalone script uses `--n-iter` and `--cv`. Both save a
model artifact that `main.py --skip-train` can load.

### Skip training (use saved model)

```bash
python src/main.py --data-dir ../../dataset --output-dir ../../output --skip-train
```

### Override prediction threshold

```bash
python src/main.py --data-dir ../../dataset --output-dir ../../output --threshold 0.6
```

### Tune blocking parameters

```bash
python src/main.py --data-dir ../../dataset --output-dir ../../output --top-k-word 80 --top-k-char 50 --min-shared-tokens 1
```

## Outputs

- `output/matching_results.tsv` - final entity matches (upload to leaderboard)
- `output/candidate_pairs.tsv` - blocking candidate set

## Approach

1. **Preprocessing**: Unicode normalization, abbreviation expansion, punctuation removal, legal suffix stripping
2. **Blocking** (3 strategies combined, per-country):
   - Word-level TF-IDF cosine on name+address (top-K)
   - Character n-gram TF-IDF on name and address (catches typos,
     transliterations, and distinctively similar premises)
   - Token blocking on name (shared token overlap)
3. **Feature Engineering**: 30+ string similarity features per pair (Levenshtein, Jaro-Winkler, Jaccard, cosine, char-ngram, number overlap, etc.)
4. **Matching**: validation-gated ensemble of class-balanced LightGBM, Extra
   Trees, and a compact MLP neural matcher. Candidate pairs for one Source 1
   entity remain together during cross-validation; model weights and the match
   cutoff are selected from out-of-fold probabilities to maximize macro
   entity-level F0.5, including singleton records. A secondary model receives
   non-zero weight only if it beats the LightGBM-only baseline in this test.
