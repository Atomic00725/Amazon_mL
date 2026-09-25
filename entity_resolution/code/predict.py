"""
Score every candidate pair on the test split with the trained model and
write the final matching_results.tsv -- the ONLY file scored on the
leaderboard during the challenge.

Usage:
    python code/predict.py --data-dir data --candidates output/candidate_pairs.tsv \
        --model output/model.joblib --out output/matching_results.tsv
"""
import argparse
import csv
import os
import sys

import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from features import pair_features  # noqa: E402


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--candidates", required=True, help="candidate_pairs.tsv from blocking.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    s1 = read_tsv(os.path.join(args.data_dir, f"{args.split}_source1.tsv"))
    s2 = read_tsv(os.path.join(args.data_dir, f"{args.split}_source2.tsv"))
    s3 = read_tsv(os.path.join(args.data_dir, f"{args.split}_source3.tsv"))
    cand_rows = read_tsv(args.candidates)

    lookup = {r["id"]: r for r in (s2 + s3)}
    candidates = {r["id"]: set(x for x in r["matches"].split(",") if x) for r in cand_rows}

    bundle = joblib.load(args.model)
    model, threshold = bundle["model"], bundle["threshold"]

    out_rows = []
    n_singletons_predicted = 0
    n_with_match = 0

    for r in s1:
        sid = r["id"]
        cand_ids = [c for c in candidates.get(sid, set()) if c in lookup]

        if not cand_ids:
            out_rows.append({"id": sid, "matches": ""})
            n_singletons_predicted += 1
            continue

        feats = [pair_features(r["name"], r["address"],
                                lookup[c]["name"], lookup[c]["address"])
                 for c in cand_ids]
        probs = model.predict_proba(np.array(feats))[:, 1]

        best = {}  # source prefix -> (score, id)
        for cid, p in zip(cand_ids, probs):
            src = cid.split("-")[0]
            if src not in best or p > best[src][0]:
                best[src] = (p, cid)

        chosen = sorted(cid for p, cid in best.values() if p >= threshold)
        if chosen:
            n_with_match += 1
        else:
            n_singletons_predicted += 1
        out_rows.append({"id": sid, "matches": ",".join(chosen)})

    write_tsv(args.out, out_rows, ["id", "matches"])
    print(f"Wrote {len(out_rows)} rows -> {args.out}")
    print(f"  predicted a match for {n_with_match} entities, "
          f"empty (no-match) for {n_singletons_predicted} entities")


if __name__ == "__main__":
    main()
