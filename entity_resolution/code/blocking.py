"""
Blocking step.

Reads source1 / source2 / source3 tsvs and produces candidate_pairs.tsv:
one row per Source 1 id, with a comma-separated shortlist of Source 2 and
Source 3 ids that MIGHT be the same business (the recall ceiling for the
whole pipeline -- a true match that never lands in the same block can
never be recovered downstream).

Multi-key blocking is used on purpose: each record is hashed into
several buckets (whole first name-token, a 4-char prefix of the first
two name-tokens, and a normalized street-name token) so that minor
typos, truncated names, or missing house numbers in one source don't
silently drop a true match's recall.
"""
import csv
import os
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address  # noqa: E402


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def record_keys(name, address, max_block_size_hint=True):
    """Return the set of blocking keys a single record participates in."""
    n = normalize_name(name)
    a = normalize_address(address)
    n_tokens = n.split()
    a_tokens = [t for t in a.split() if not t.isdigit()]

    keys = set()
    if n_tokens:
        keys.add(f"n0:{n_tokens[0]}")
        keys.add(f"n0p:{n_tokens[0][:4]}")
        if len(n_tokens) > 1:
            keys.add(f"n1p:{n_tokens[1][:4]}")
    if a_tokens:
        # drop trailing street-type abbreviations (st/ave/rd/...) so the
        # key lands on the actual street or place name
        street_type_abbrs = {"st", "ave", "rd", "dr", "blvd", "ln", "ct",
                              "pl", "cir", "hwy"}
        core = [t for t in a_tokens if t not in street_type_abbrs]
        if core:
            keys.add(f"a0:{core[0]}")
    return keys


def build_blocks(s1_rows, s2_rows, s3_rows):
    block_index = defaultdict(lambda: {"s2": [], "s3": []})

    for r in s2_rows:
        for k in record_keys(r["name"], r["address"]):
            block_index[k]["s2"].append(r["id"])
    for r in s3_rows:
        for k in record_keys(r["name"], r["address"]):
            block_index[k]["s3"].append(r["id"])

    candidates = {}  # s1_id -> set of candidate ids
    for r in s1_rows:
        cand = set()
        for k in record_keys(r["name"], r["address"]):
            entry = block_index.get(k)
            if not entry:
                continue
            cand.update(entry["s2"])
            cand.update(entry["s3"])
        candidates[r["id"]] = cand
    return candidates


def run(data_dir, split, out_path):
    s1 = read_tsv(os.path.join(data_dir, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(data_dir, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(data_dir, f"{split}_source3.tsv"))

    candidates = build_blocks(s1, s2, s3)

    rows = []
    total_candidates = 0
    for r in s1:
        ids = sorted(candidates.get(r["id"], []))
        total_candidates += len(ids)
        rows.append({"id": r["id"], "matches": ",".join(ids)})

    write_tsv(out_path, rows, ["id", "matches"])
    avg = total_candidates / max(1, len(s1))
    print(f"[{split}] {len(s1)} Source-1 records -> avg {avg:.1f} candidates each "
          f"-> {out_path}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.data_dir, args.split, args.out)
