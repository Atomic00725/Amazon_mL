"""
Pairwise feature extraction for (Source-1 record, candidate record) pairs.
Pure string/structural similarity -- NO external lookups, per the rules
("no external data lookup: external databases, APIs, and geocoding
services are strictly prohibited").
"""
import sys
import os

from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address  # noqa: E402

FEATURE_NAMES = [
    "name_ratio", "name_token_sort", "name_token_set", "name_partial",
    "name_first_token_exact", "name_len_diff",
    "addr_ratio", "addr_token_sort", "addr_token_set", "addr_partial",
    "addr_num_match", "addr_num_present_both",
    "shared_token_count", "shared_token_jaccard",
]


def _numbers(text):
    return set(t for t in text.split() if t.isdigit())


def pair_features(name_a, addr_a, name_b, addr_b):
    na, nb = normalize_name(name_a), normalize_name(name_b)
    aa, ab = normalize_address(addr_a), normalize_address(addr_b)

    name_ratio = fuzz.ratio(na, nb) / 100.0
    name_token_sort = fuzz.token_sort_ratio(na, nb) / 100.0
    name_token_set = fuzz.token_set_ratio(na, nb) / 100.0
    name_partial = fuzz.partial_ratio(na, nb) / 100.0
    name_first_exact = 1.0 if (na.split() and nb.split() and na.split()[0] == nb.split()[0]) else 0.0
    name_len_diff = abs(len(na) - len(nb)) / (max(len(na), len(nb), 1))

    addr_ratio = fuzz.ratio(aa, ab) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(aa, ab) / 100.0
    addr_token_set = fuzz.token_set_ratio(aa, ab) / 100.0
    addr_partial = fuzz.partial_ratio(aa, ab) / 100.0

    nums_a, nums_b = _numbers(aa), _numbers(ab)
    both_present = 1.0 if (nums_a and nums_b) else 0.0
    num_match = 1.0 if (nums_a and nums_b and nums_a & nums_b) else 0.0

    tokens_a = set(na.split()) | set(aa.split())
    tokens_b = set(nb.split()) | set(ab.split())
    shared = tokens_a & tokens_b
    union = tokens_a | tokens_b
    shared_count = len(shared)
    jaccard = len(shared) / len(union) if union else 0.0

    return [
        name_ratio, name_token_sort, name_token_set, name_partial,
        name_first_exact, name_len_diff,
        addr_ratio, addr_token_sort, addr_token_set, addr_partial,
        num_match, both_present,
        shared_count, jaccard,
    ]
