"""
Reference implementation of the leaderboard metric described in the
brief: macro F-beta (beta=0.5, i.e. precision-weighted), computed
PER SOURCE-1 ENTITY and averaged.

For a given Source-1 id:
  predicted = set of ids you output for it
  truth     = set of ids ground truth says match it

  if truth is empty (a singleton):
      score = 1.0 if predicted is empty else 0.0
  else:
      precision = |predicted & truth| / |predicted|   (1.0 if predicted empty)
      recall    = |predicted & truth| / |truth|
      score     = F_beta(precision, recall)

The final score is the mean over all Source-1 ids. This mirrors the
brief's stated rules: a false merge (hurts precision) costs roughly 2x
a miss (hurts recall) under beta=0.5, and predicting an empty list for
a true singleton scores a full 1.0.
"""


def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(predicted, truth, beta=0.5):
    predicted = set(x for x in predicted if x)
    truth = set(x for x in truth if x)

    if not truth:
        return 1.0 if not predicted else 0.0

    if not predicted:
        return 0.0  # precision undefined -> 0 recall achieved, F = 0

    overlap = len(predicted & truth)
    precision = overlap / len(predicted)
    recall = overlap / len(truth)
    return f_beta(precision, recall, beta)


def macro_f_beta(pred_dict, truth_dict, beta=0.5):
    """
    pred_dict, truth_dict: {source1_id: set_of_ids}
    Scores every id present in truth_dict (predictions default to empty
    set if missing).
    """
    scores = []
    for sid, truth in truth_dict.items():
        pred = pred_dict.get(sid, set())
        scores.append(score_entity(pred, truth, beta))
    return sum(scores) / len(scores) if scores else 0.0
