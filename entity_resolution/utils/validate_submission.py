"""
Validate matching_results.tsv (and optionally candidate_pairs.tsv) before
you submit -- catches the formatting mistakes that silently zero out a
leaderboard score.

Usage:
    python utils/validate_submission.py \
        --data-dir data --split test \
        --matching output/matching_results.tsv \
        --candidates output/candidate_pairs.tsv
"""
import argparse
import csv
import os
import sys


def read_tsv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = list(reader)
    return header, rows


def fail(msg, errors):
    errors.append(msg)


def validate_matching_file(path, expected_ids, valid_candidate_ids, label):
    errors = []
    warnings = []

    if not os.path.exists(path):
        return [f"[{label}] File not found: {path}"], warnings

    with open(path, newline="") as f:
        first_line = f.readline()
    if "\t" not in first_line:
        fail(f"[{label}] File does not look tab-separated (no \\t found in header line). "
             f"Read/write with sep='\\t'.", errors)

    header, rows = read_tsv(path)
    if header != ["id", "matches"]:
        fail(f"[{label}] Header must be exactly ['id', 'matches'], got {header}", errors)

    seen_ids = set()
    dupe_ids = set()
    bad_rows = 0
    unknown_ref_count = 0

    for i, row in enumerate(rows, start=2):
        if len(row) < 2:
            # allow a trailing empty "matches" field to collapse to 1 column
            if len(row) == 1:
                row = [row[0], ""]
            else:
                bad_rows += 1
                continue
        rid, matches = row[0], row[1]

        if rid in seen_ids:
            dupe_ids.add(rid)
        seen_ids.add(rid)

        if matches.strip() != matches:
            fail(f"[{label}] Row {i}: matches field has leading/trailing whitespace: {matches!r}", errors)

        if matches:
            for m in matches.split(","):
                if m != m.strip():
                    fail(f"[{label}] Row {i}: candidate id has stray whitespace: {m!r}", errors)
                if valid_candidate_ids is not None and m.strip() not in valid_candidate_ids:
                    unknown_ref_count += 1

    if bad_rows:
        fail(f"[{label}] {bad_rows} malformed row(s) (wrong column count)", errors)
    if dupe_ids:
        fail(f"[{label}] Duplicate id(s) found, e.g. {list(dupe_ids)[:5]}", errors)

    missing = expected_ids - seen_ids
    extra = seen_ids - expected_ids
    if missing:
        fail(f"[{label}] Missing {len(missing)} required Source-1 id(s), "
             f"e.g. {list(missing)[:5]}. Every Source-1 test entity needs a row, "
             f"even if 'matches' is empty.", errors)
    if extra:
        warnings.append(f"[{label}] {len(extra)} row(s) have an id not present in Source-1 test set, "
                         f"e.g. {list(extra)[:5]}")
    if unknown_ref_count:
        fail(f"[{label}] {unknown_ref_count} referenced candidate id(s) don't exist in "
             f"Source-2/Source-3 test data.", errors)

    return errors, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidates", default=None)
    args = ap.parse_args()

    def load_ids(path):
        with open(path, newline="") as f:
            return set(r["id"] for r in csv.DictReader(f, delimiter="\t"))

    s1_path = os.path.join(args.data_dir, f"{args.split}_source1.tsv")
    s2_path = os.path.join(args.data_dir, f"{args.split}_source2.tsv")
    s3_path = os.path.join(args.data_dir, f"{args.split}_source3.tsv")

    if not os.path.exists(s1_path):
        print(f"Could not find {s1_path} -- point --data-dir at the folder holding "
              f"{args.split}_source1/2/3.tsv")
        sys.exit(2)

    expected_ids = load_ids(s1_path)
    valid_candidate_ids = load_ids(s2_path) | load_ids(s3_path)

    all_errors, all_warnings = [], []

    errs, warns = validate_matching_file(args.matching, expected_ids, valid_candidate_ids, "matching_results")
    all_errors += errs
    all_warnings += warns

    if args.candidates:
        errs, warns = validate_matching_file(args.candidates, expected_ids, valid_candidate_ids, "candidate_pairs")
        all_errors += errs
        all_warnings += warns

    print(f"Checked against {len(expected_ids)} Source-1 {args.split} entities.\n")

    if all_warnings:
        print("WARNINGS:")
        for w in all_warnings:
            print("  -", w)
        print()

    if all_errors:
        print("FAILED -- fix these before submitting:")
        for e in all_errors:
            print("  -", e)
        sys.exit(1)
    else:
        print("PASSED -- file format looks correct.")


if __name__ == "__main__":
    main()
