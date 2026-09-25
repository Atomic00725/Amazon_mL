# Business Entity Resolution — Starter Pipeline (Amazon ML Challenge 2026)

This is a **complete, working, end-to-end** solution for the challenge
described in the brief. It currently runs on **synthetic data** that
mimics the exact schema described in the brief, so you can see it work
today. Swap in the real files and it runs unchanged.

## Quick start
```bash
pip install pandas scikit-learn rapidfuzz joblib --break-system-packages

# 1. (only needed for the demo) generate synthetic data matching the schema
python code/data_gen.py

# 2. run everything: blocking -> train -> blocking(test) -> predict -> validate
python code/pipeline.py --data-dir data --out-dir output
```

That produces:
- `output/candidate_pairs.tsv` — the audited blocking shortlist
- `output/matching_results.tsv` — the file scored on the leaderboard
- `output/model.joblib` — the trained model

## Switching to the REAL challenge data
1. Drop the real files into `data/` using these exact names (same schema
   used throughout the code — just tab-separated `id`, `name`, `address`
   columns for source files, and `id`, `matches` for ground truth):
   - `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`
   - `train_ground_truth.tsv`
   - `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`
2. Delete or skip `code/data_gen.py` — don't run it, it will overwrite
   these with synthetic data.
3. Re-run: `python code/pipeline.py --data-dir data --out-dir output`

If the real column names differ even slightly (e.g. `business_name`
instead of `name`), update the `fieldnames` reads in `code/blocking.py`
/ `code/train.py` / `code/predict.py` — everything else (normalization,
features, model, metric) is schema-agnostic.

## Where to focus your improvements
1. **Blocking recall** (`code/blocking.py`, `code/normalize.py`) — this
   sets your ceiling. Check the printed "avg candidates per entity" and
   the recall-ceiling snippet in `code/metric.py`-based validation; if
   real names/addresses have patterns the current keys miss (e.g.
   transliteration, abbreviated states, alternate street naming), add a
   blocking key for it.
2. **Features** (`code/features.py`) — add anything else derivable from
   the raw text without external lookups (e.g. detected suite/unit
   numbers, normalized ZIP/postal code if present, business-type token
   sets).
3. **Threshold / decision rule** (`code/train.py`, `code/predict.py`) —
   currently "best per source above a global threshold, threshold tuned
   for macro F0.5." Consider per-block or per-source thresholds if
   precision/recall trade off differently across sources.
4. **Model** — a RandomForest is the baseline; gradient boosting
   (XGBoost/LightGBM) or a small learned embedding + cosine similarity
   often pushes precision further with the same features.

## File map
```
data/           challenge data (synthetic here — replace with the real files)
code/
  data_gen.py    synthetic data generator (demo only)
  normalize.py   shared text normalization (used by every stage)
  blocking.py    candidate-pair generation
  features.py    pairwise similarity features
  metric.py      reference implementation of the leaderboard metric
  train.py       trains the classifier + tunes the decision threshold
  predict.py     produces matching_results.tsv
  pipeline.py    runs all of the above in one command
utils/
  validate_submission.py   format checker to run before you submit
output/          candidate_pairs.tsv, matching_results.tsv, model.joblib
Documentation_template.md  filled-in methodology writeup
```
