# Colab training guide

The supplied files contain roughly 2.2 million training Source-1 records and
10.3 million training retrieval records. Use a **high-RAM Colab runtime**;
standard free-RAM sessions are not suitable for the complete run. GPU is useful
for experimentation but this reproducible LightGBM / Extra Trees / MLP pipeline
also runs on CPU.

## 1. Prepare the runtime

Upload the supplied data ZIP to Drive. Then run this in the first Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/Farhancoader/amazommlzero.git /content/amazommlzero
%cd /content/amazommlzero
!pip install -q -r code/business_entity_resolution/requirements.txt
```

Then extract the provided data ZIP. The following creates the exact layout
expected by the code:

```python
!unzip -q '/content/drive/MyDrive/6ab10eb3b23ba_student_resource.zip' -d /content/archive
!mv /content/archive/student_resource/dataset /content/amazommlzero/dataset
```

## 2. Run a calibration experiment

Start with a three-fold run. It trains LightGBM, Extra Trees, and the neural MLP
on the same candidate pairs, then retains a non-zero ensemble weight for a
secondary model only if grouped out-of-fold macro F0.5 improves.

```python
%cd /content/amazommlzero/code/business_entity_resolution
!python src/main.py \
  --data-dir ../../dataset \
  --output-dir ../../output \
  --model-dir ./models \
  --cv-folds 3 \
  --search-iterations 10 \
  --top-k-word 60 \
  --top-k-char 30 \
  --top-k-addr-char 15 \
  --min-shared-tokens 1
```

The sparse top-N TF-IDF implementation is required for this data scale. It
never materializes an S1-by-S2/S3 dense similarity matrix. Candidate generation
is the main recall lever: increase one top-K setting at a time only when the
reported training blocking recall rises enough to justify the extra candidate
pairs and runtime.

## 3. Validate and submit

The full command writes:

```text
/content/amazommlzero/output/matching_results.tsv
/content/amazommlzero/output/candidate_pairs.tsv
```

Validate the two files before upload:

```python
%cd /content/amazommlzero/student_resource
!python utils/validate_submission.py \
  --matching ../output/matching_results.tsv \
  --candidate ../output/candidate_pairs.tsv \
  --test-dir ../dataset/test
```

Do not use business lookup APIs, geocoders, or external business data. This
pipeline learns only from the supplied training labels and record text.
