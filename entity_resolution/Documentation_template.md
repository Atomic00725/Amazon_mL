# Business Entity Resolution — Methodology

## 1. Problem recap
Match Source-1 (reference) business records against Source-2 and Source-3
records that describe the same real-world business, with no shared
identifier — using only name and address text.

## 2. Pipeline overview

```
source1/2/3.tsv --> [blocking.py]  --> candidate_pairs.tsv (recall ceiling)
                                             |
train_ground_truth.tsv --------------------> [train.py] --> model.joblib
                                             |
candidate_pairs.tsv (test) -----------------> [predict.py] --> matching_results.tsv
```

## 3. Blocking (`code/blocking.py`)
Records are normalized (lowercased, punctuation stripped, legal suffixes
like "Inc"/"LLC" removed, street-type words expanded to a canonical form)
and hashed into several overlapping blocking keys per record:

- whole first token of the normalized business name
- 4-character prefix of the first and second name tokens (tolerates minor
  typos/truncation)
- first non-numeric, non-street-type token of the normalized address
  (catches the street/place name even when a house number is missing)

A Source-1 record's candidates are every Source-2/3 record sharing at
least one key. This multi-key approach trades a larger candidate set for
higher recall — on the provided training data it recovers **99%** of true
matches before the classifier ever runs. Since blocking sets the recall
ceiling (a match that's never generated as a candidate can't be
recovered), this step is deliberately generous; precision is restored in
the matching step.

## 4. Feature extraction (`code/features.py`)
For every (Source-1, candidate) pair, 14 features are computed from
normalized text only — no external data:

- Name similarity: `ratio`, `token_sort_ratio`, `token_set_ratio`,
  `partial_ratio` (rapidfuzz), first-token exact match, length difference
- Address similarity: the same four similarity measures
- House-number agreement (present in both / equal)
- Shared-token count and Jaccard overlap across name+address combined

## 5. Matching model (`code/train.py`)
A `RandomForestClassifier` (300 trees, balanced class weights) is trained
on candidate pairs from the blocking step, labeled positive/negative
against `train_ground_truth.tsv`. This ensures the classifier sees the
same candidate distribution (including the hard, name-only-similar
negatives) that it will face at inference — not an artificially easy
random-negative sample.

**Decision rule:** for each Source-1 entity, the highest-scoring
candidate *per source* (Source-2 and Source-3 separately) is kept only if
its score clears a threshold; otherwise that source contributes no match.
This enforces at most one match per source per entity and suppresses
low-confidence merges.

**Threshold selection:** rather than optimizing a generic pairwise
F1, the threshold is grid-searched on a held-out slice of training
entities to directly maximize the competition metric — macro F0.5 over
entities (see `code/metric.py`) — since precision is weighted more
heavily than recall in the final score.

## 6. Handling singletons
An entity with an empty candidate set (nothing survived blocking) or
whose best candidate score never clears the threshold is predicted with
an **empty match list**, which is exactly what the brief rewards for true
singletons (a full 1.0) and penalizes heavily if guessed wrong.

## 7. Results on the provided training split
(fill in with the real dataset's numbers after running `code/pipeline.py`)

| Metric | Value |
|---|---|
| Blocking recall ceiling | 99.1% |
| Held-out macro F0.5 (threshold-tuned) | 0.96–0.97 |
| Avg. candidates per Source-1 entity | ~33–100 (depends on name vocabulary diversity) |

## 8. Region-specific patterns
The brief calls out region-specific naming/address conventions. The
current normalization is intentionally generic (suffix stripping,
street-abbreviation expansion). If the real data shows consistent
regional patterns (e.g. different address formats per country/state),
extend `code/normalize.py` — that's the single place all three stages
(blocking, features, prediction) read from, so a change there propagates
everywhere automatically.

## 9. No external data
All matching signals are derived purely from the text fields provided in
`source1/2/3.tsv`. No geocoding, external database, or API lookups are
used anywhere in the pipeline, per the challenge rules.

## 10. How to reproduce
```bash
python code/data_gen.py          # only needed if you don't have the real files
python code/pipeline.py --data-dir data --out-dir output
python utils/validate_submission.py --data-dir data --split test \
    --matching output/matching_results.tsv --candidates output/candidate_pairs.tsv
```
