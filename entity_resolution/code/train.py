"""
Train the candidate-pair classifier and pick a decision threshold that
maximizes the ACTUAL leaderboard metric (macro F0.5 over Source-1
entities) on a held-out slice of the training entities -- not a generic
pairwise F1, since the two are not the same target.

Usage:
    python code/train.py --data-dir data --candidates output/train_candidate_pairs.tsv \
        --model-out output/model.joblib
"""
import argparse
import csv
import os
import sys
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from features import pair_features, FEATURE_NAMES  # noqa: E402
from metric import macro_f_beta  # noqa: E402

random.seed(0)


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_lookup(rows):
    return {r["id"]: r for r in rows}


def load_match_dict(rows):
    out = {}
    for r in rows:
        ids = set(x for x in r["matches"].split(",") if x)
        out[r["id"]] = ids
    return out


def build_pair_dataset(s1_rows, lookup, candidates, truth, ids_subset=None):
    X, y, meta = [], [], []
    for r in s1_rows:
        sid = r["id"]
        if ids_subset is not None and sid not in ids_subset:
            continue
        true_matches = truth.get(sid, set())
        for cid in candidates.get(sid, set()):
            crec = lookup.get(cid)
            if crec is None:
                continue
            feats = pair_features(r["name"], r["address"], crec["name"], crec["address"])
            X.append(feats)
            y.append(1 if cid in true_matches else 0)
            meta.append((sid, cid))
    return np.array(X, dtype=float), np.array(y, dtype=int), meta


def predict_for_entities(s1_rows, lookup, candidates, model, ids_subset, threshold):
    """Apply the 'best candidate per source, above threshold' decision rule."""
    preds = {}
    for r in s1_rows:
        sid = r["id"]
        if ids_subset is not None and sid not in ids_subset:
            continue
        cand_ids = list(candidates.get(sid, set()))
        if not cand_ids:
            preds[sid] = set()
            continue
        feats = [pair_features(r["name"], r["address"],
                                lookup[c]["name"], lookup[c]["address"])
                 for c in cand_ids if c in lookup]
        cand_ids = [c for c in cand_ids if c in lookup]
        if not feats:
            preds[sid] = set()
            continue
        probs = model.predict_proba(np.array(feats))[:, 1]

        best = {}  # source prefix -> (score, id)
        for cid, p in zip(cand_ids, probs):
            src = cid.split("-")[0]
            if src not in best or p > best[src][0]:
                best[src] = (p, cid)

        chosen = {cid for p, cid in best.values() if p >= threshold}
        preds[sid] = chosen
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--candidates", required=True, help="train candidate_pairs.tsv from blocking.py")
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    s1 = read_tsv(os.path.join(args.data_dir, "train_source1.tsv"))
    s2 = read_tsv(os.path.join(args.data_dir, "train_source2.tsv"))
    s3 = read_tsv(os.path.join(args.data_dir, "train_source3.tsv"))
    truth_rows = read_tsv(os.path.join(args.data_dir, "train_ground_truth.tsv"))
    cand_rows = read_tsv(args.candidates)

    lookup = {**load_lookup(s2), **load_lookup(s3)}
    truth = load_match_dict(truth_rows)
    candidates = load_match_dict(cand_rows)

    all_ids = [r["id"] for r in s1]
    random.shuffle(all_ids)
    n_val = int(len(all_ids) * args.val_frac)
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])

    print(f"Train entities: {len(train_ids)}  Val entities: {len(val_ids)}")

    X_train, y_train, _ = build_pair_dataset(s1, lookup, candidates, truth, train_ids)
    print(f"Training pairs: {len(y_train)}  positive rate: {y_train.mean():.3f}")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=0,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ---- Tune the decision threshold on the held-out entities against
    # ---- the REAL competition metric (macro F0.5), not pairwise F1.
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        preds = predict_for_entities(s1, lookup, candidates, clf, val_ids, t)
        val_truth = {sid: truth.get(sid, set()) for sid in val_ids}
        score = macro_f_beta(preds, val_truth, beta=0.5)
        if score > best_score:
            best_score, best_t = score, t

    print(f"Best threshold: {best_t:.2f}  -> val macro F0.5: {best_score:.4f}")

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                          key=lambda x: -x[1])
    print("Top features:")
    for name, imp in importances[:6]:
        print(f"  {name:25s} {imp:.3f}")

    joblib.dump({"model": clf, "threshold": float(best_t)}, args.model_out)
    print("Saved model ->", args.model_out)


if __name__ == "__main__":
    main()
