AMAZON ML CHALLENGE 2026 - ENTITY RESOLUTION HANDOFF
====================================================

Purpose
-------
This repository matches each Source 1 business to every corresponding Source 2
and Source 3 business. It produces the two required TSV files:

  output/matching_results.tsv   <- upload this file to the leaderboard
  output/candidate_pairs.tsv    <- auditable final candidate set

The evaluator uses macro F0.5. Precision matters more than recall: a false
business merge is more costly than a missed match. Predicting no match is the
right answer for a singleton.

Repository layout
-----------------
  code/business_entity_resolution/src/main.py       end-to-end entry point
  code/business_entity_resolution/src/blocking.py   sparse candidate retrieval
  code/business_entity_resolution/src/features.py   pairwise similarity features
  code/business_entity_resolution/src/train.py      ensemble + OOF calibration
  code/business_entity_resolution/COLAB.md          copy/paste Colab workflow
  student_resource/utils/validate_submission.py     submission format validator

The dataset is intentionally NOT in Git. It must have this layout at the
repository root after extraction:

  dataset/train/train_source1.tsv
  dataset/train/train_source2.tsv
  dataset/train/train_source3.tsv
  dataset/train/train_ground_truth.tsv
  dataset/test/test_source1.tsv
  dataset/test/test_source2.tsv
  dataset/test/test_source3.tsv

Important data scale
--------------------
The supplied training set has 2,206,821 Source 1 records, 10,320,219 Source
2/3 records, and 7,638,365 labelled match links. Use a high-RAM Colab runtime.
Do NOT replace sparse retrieval with a dense S1-by-target cosine matrix: it
will run out of memory.

Quick start in Colab
--------------------
1. Open code/business_entity_resolution/COLAB.md and follow it exactly.
2. Install dependencies:

     pip install -r code/business_entity_resolution/requirements.txt

3. From code/business_entity_resolution, run the first full experiment:

     python src/main.py ^
       --data-dir ../../dataset ^
       --output-dir ../../output ^
       --model-dir ./models ^
       --cv-folds 3 ^
       --search-iterations 10 ^
       --top-k-word 60 ^
       --top-k-char 30 ^
       --top-k-addr-char 15 ^
       --min-shared-tokens 1

   In a Linux/Colab shell, use a backslash (\) rather than a caret (^) for
   line continuation, or put the command on one line.

4. Validate before submitting:

     cd ../../student_resource
     python utils/validate_submission.py ^
       --matching ../output/matching_results.tsv ^
       --candidate ../output/candidate_pairs.tsv ^
       --test-dir ../dataset/test

How the model works
-------------------
1. Text normalization standardizes casing, Unicode, common legal suffixes,
   business/address abbreviations, punctuation, and country labels.
2. Candidate generation runs only within the same normalized country. It unions
   word TF-IDF over name+address, character TF-IDF over names, character TF-IDF
   over addresses, and rare shared-name-token blocks. sparse-dot-topn retains
   just the best candidates per source record without allocating a dense matrix.
3. The matcher calculates name/address edit, Jaro-Winkler, token overlap,
   containment, n-gram, number, and structural features. RapidFuzz accelerates
   expensive string comparisons.
4. Three models are trained on those features: tuned class-balanced LightGBM,
   Extra Trees, and a compact class-balanced neural MLP.
5. Model weights and the final probability threshold are selected only from
   grouped out-of-fold predictions. All pairs for one S1 record stay in one
   fold. The exact macro entity-level F0.5 scorer, including singletons, chooses
   the final blend and threshold. A secondary model gets zero weight if it does
   not beat LightGBM in this held-out evaluation.

Safe tuning procedure
---------------------
Change one thing at a time and record: blocking recall, number of candidate
pairs, runtime/RAM, and validation macro F0.5.

  * First run: use the command above. It is the accuracy/runtime starting point.
  * If blocking recall is below the desired ceiling, raise ONE top-K at a time
    (word to 80, name-char to 40, or address-char to 25). More candidates may
    improve recall but also create more false positives and much more work.
  * If RAM is tight, lower address-char first; it is the most expensive pass.
    Never reintroduce a dense similarity matrix.
  * Do not manually set a threshold based on intuition. The pipeline's OOF
    threshold is deliberately precision-oriented for F0.5. Compare overrides
    only on a proper held-out validation run.
  * Do not add country one-hot features or hard-code US/India. France appears
    only in test, so country handling must remain open-set string based.
  * Any model change must be evaluated with S1-grouped folds. Pair-level random
    splits leak business context and produce over-optimistic results.

Rules for a future AI agent
---------------------------
  * Use only the supplied TSV data. Never call external business lookup,
    geocoding, registry, or enrichment APIs; that disqualifies the submission.
  * Preserve all Source 1 rows in both output files. A final match must be a
    member of that S1 record's candidate list.
  * Treat blocking recall as the hard ceiling on match recall. Measure it before
    spending time on a new classifier.
  * Make changes in small, reproducible commits. Do not commit dataset/, output/
    or models/; .gitignore intentionally excludes them.
  * Run the validator after every final-output run. It catches structural
    failures but does not calculate leaderboard quality.
  * Document the final validation score, blocking settings, blend weights, and
    threshold in Documentation_template.md before the final package is created.

Known limitations and next best experiments
-------------------------------------------
The supplied sparse lexical retrievers are intentionally data-only and auditable.
The strongest next experiment is usually candidate-generation tuning, not adding
more classifiers. If a neural architecture is changed, keep the MLP only when
the grouped OOF blend assigns it a non-zero weight. Do not claim an improvement
until the complete validation run finishes and the score is recorded.
