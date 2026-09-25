"""
Runs the full pipeline end to end:

  1. Block  : data/train_source[1-3].tsv       -> output/train_candidate_pairs.tsv
  2. Train  : output/train_candidate_pairs.tsv + train_ground_truth.tsv -> output/model.joblib
  3. Block  : data/test_source[1-3].tsv        -> output/candidate_pairs.tsv   (submitted)
  4. Predict: output/candidate_pairs.tsv + model.joblib -> output/matching_results.tsv (submitted)
  5. Validate the two output files.

Usage:
    python code/pipeline.py --data-dir data --out-dir output
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "output"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    py = sys.executable

    train_cand = os.path.join(args.out_dir, "train_candidate_pairs.tsv")
    test_cand = os.path.join(args.out_dir, "candidate_pairs.tsv")
    model_path = os.path.join(args.out_dir, "model.joblib")
    result_path = os.path.join(args.out_dir, "matching_results.tsv")

    run([py, os.path.join(HERE, "blocking.py"), "--data-dir", args.data_dir,
         "--split", "train", "--out", train_cand])

    run([py, os.path.join(HERE, "train.py"), "--data-dir", args.data_dir,
         "--candidates", train_cand, "--model-out", model_path])

    run([py, os.path.join(HERE, "blocking.py"), "--data-dir", args.data_dir,
         "--split", "test", "--out", test_cand])

    run([py, os.path.join(HERE, "predict.py"), "--data-dir", args.data_dir,
         "--candidates", test_cand, "--model", model_path,
         "--out", result_path, "--split", "test"])

    run([py, os.path.join(HERE, "..", "utils", "validate_submission.py"),
         "--data-dir", args.data_dir, "--split", "test",
         "--matching", result_path, "--candidates", test_cand])

    print("\nDone. Submit:", result_path)
    print("Final package should also include:", test_cand)


if __name__ == "__main__":
    main()
