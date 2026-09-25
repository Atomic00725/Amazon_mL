"""
Shared text-normalization helpers used by blocking, feature extraction,
and prediction. Keeping this in one module guarantees train/test/blocking
all normalize identically.
"""
import re

LEGAL_SUFFIXES = [
    "incorporated", "inc", "llc", "ltd", "limited", "co", "corp",
    "corporation", "company", "llp", "plc",
]

STREET_ABBR = {
    "street": "st", "avenue": "ave", "road": "rd", "drive": "dr",
    "boulevard": "blvd", "lane": "ln", "court": "ct", "place": "pl",
    "circle": "cir", "highway": "hwy",
}

_word_re = re.compile(r"[a-z0-9]+")


def _words(text):
    return _word_re.findall(text.lower())


def normalize_name(name):
    """Lowercase, strip punctuation, drop common legal suffixes."""
    if not name:
        return ""
    words = _words(name)
    words = [w for w in words if w not in LEGAL_SUFFIXES]
    return " ".join(words)


def normalize_address(address):
    """Lowercase, strip punctuation, expand street-type abbreviations."""
    if not address:
        return ""
    words = _words(address)
    words = [STREET_ABBR.get(w, w) for w in words]
    return " ".join(words)


def name_first_token(name_norm):
    words = name_norm.split()
    return words[0] if words else ""


def address_city_guess(address_norm):
    """
    Heuristic: the city is usually the last 1-2 alphabetic tokens after
    a comma in the raw address; since we've already stripped punctuation,
    fall back to the trailing non-numeric words.
    """
    words = address_norm.split()
    tail = [w for w in words if not w.isdigit()]
    return " ".join(tail[-2:]) if tail else ""


def blocking_key(name, address):
    """
    Cheap key used to bucket records: first token of the normalized
    business name + a coarse guess at the city/locality. Two records
    only become comparison candidates if they land in the same block,
    so this key controls the recall ceiling (see brief: 'you cannot
    match a record you never consider').
    """
    n = normalize_name(name)
    a = normalize_address(address)
    return f"{name_first_token(n)}|{address_city_guess(a)}"
