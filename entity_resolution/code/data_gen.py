"""
Synthetic data generator for the Business Entity Resolution challenge.

Produces data in the SAME shape described in the challenge brief:
  - source1.tsv  (reference, deduplicated)
  - source2.tsv  (vendor A format)
  - source3.tsv  (vendor B format)
  - train_ground_truth.tsv  (id -> comma separated list of matching ids)
  - test_source1.tsv / test_source2.tsv / test_source3.tsv (no labels)

This is ONLY a stand-in so the pipeline is runnable end-to-end today.
Swap these files for the real challenge files (same column names) and
everything downstream (blocking / features / train / predict) works
unchanged.
"""
import random
import csv
import os

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

CITIES = [
    ("San Jose", "CA"), ("Austin", "TX"), ("Reno", "NV"), ("Boise", "ID"),
    ("Fargo", "ND"), ("Tulsa", "OK"), ("Dayton", "OH"), ("Provo", "UT"),
]

STREET_TYPES = ["St", "Ave", "Rd", "Dr", "Blvd", "Ln"]
STREET_TYPES_LONG = {"St": "Street", "Ave": "Avenue", "Rd": "Road",
                      "Dr": "Drive", "Blvd": "Boulevard", "Ln": "Lane"}

BIZ_WORDS = ["Acme", "Delta", "Zen", "Bright", "Kappa", "Summit", "Cedar",
             "Falcon", "River", "Golden", "Blue", "North", "Union", "Vertex",
             "Harbor", "Maple", "Silver", "Atlas", "Orbit", "Pioneer"]
BIZ_TYPES = ["Robotics", "Foods", "Traders", "Cafe", "Motors", "Bakery",
             "Logistics", "Electronics", "Consulting", "Textiles", "Farms",
             "Media", "Hardware", "Apparel", "Systems"]
SUFFIXES = ["Inc.", "Incorporated", "LLC", "Co", "Ltd", ""]


def rand_business_name():
    return f"{random.choice(BIZ_WORDS)} {random.choice(BIZ_TYPES)}"


def rand_address(city, state):
    num = random.randint(1, 9999)
    street = f"{random.choice(BIZ_WORDS)} {random.choice(STREET_TYPES)}"
    return num, street, city, state


def noisy_name(name, level):
    """Apply vendor-style noise to a business name."""
    n = name
    if level == 0:
        # reference: sometimes has a clean suffix
        if random.random() < 0.4:
            n = f"{n} {random.choice(['Inc.', 'LLC', 'Co'])}"
        return n
    if level == 1:
        # vendor A: verbose suffixes, extra whitespace
        if random.random() < 0.7:
            n = f"{n} {random.choice(['Incorporated', 'LLC', 'Limited'])}"
        return n
    # vendor B: often drops suffix, sometimes truncates / typos
    if random.random() < 0.3:
        n = n.split(" ")[0]  # truncate to first word only
    if random.random() < 0.15 and len(n) > 4:
        i = random.randint(1, len(n) - 2)
        n = n[:i] + n[i + 1] + n[i] + n[i + 2:]  # swap two chars (typo)
    return n


def noisy_address(num, street, street_type, city, state, level):
    st_disp = street_type if random.random() < 0.5 else STREET_TYPES_LONG[street_type]
    if level == 0:
        return f"{num} {street} {st_disp}, {city}"
    if level == 1:
        return f"{num} {street} {st_disp}, {city} {state}"
    # vendor B: sometimes uses a landmark instead of a house number
    if random.random() < 0.35:
        return f"Nr. {random.choice(['City Hall', 'Central Park', 'Main Square', 'Bus Depot'])}, {city}"
    return f"{num} {street} {st_disp}, {city}"


def build_entities(n=400):
    entities = []
    for i in range(n):
        name = rand_business_name()
        city, state = random.choice(CITIES)
        num, street, city, state = rand_address(city, state)
        street_type = random.choice(STREET_TYPES)
        entities.append({
            "id": i,
            "name": name,
            "num": num,
            "street": street,
            "street_type": street_type,
            "city": city,
            "state": state,
        })
    return entities


def write_tsv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def generate_split(entities, split_name, presence_probs=(1.0, 0.85, 0.85),
                    singleton_rate=0.12):
    """
    Build source1/2/3 rows + ground truth for a set of entities.
    Some entities are singletons (appear only in source 1, i.e. no match
    anywhere) to mirror the brief's note about singletons.
    """
    n = len(entities)
    is_singleton = [random.random() < singleton_rate for _ in range(n)]

    s1_rows, s2_rows, s3_rows = [], [], []
    gt_rows = []

    s1_id_counter = 1
    s2_id_counter = 1
    s3_id_counter = 1

    for idx, e in enumerate(entities):
        s1_id = f"S1-{s1_id_counter:06d}"
        s1_id_counter += 1
        name0 = noisy_name(e["name"], 0)
        addr0 = noisy_address(e["num"], e["street"], e["street_type"], e["city"], e["state"], 0)
        s1_rows.append({"id": s1_id, "name": name0, "address": addr0})

        matched_s2, matched_s3 = [], []

        if not is_singleton[idx]:
            if random.random() < presence_probs[1]:
                s2_id = f"S2-{s2_id_counter:06d}"
                s2_id_counter += 1
                name1 = noisy_name(e["name"], 1)
                addr1 = noisy_address(e["num"], e["street"], e["street_type"], e["city"], e["state"], 1)
                s2_rows.append({"id": s2_id, "name": name1, "address": addr1})
                matched_s2.append(s2_id)

            if random.random() < presence_probs[2]:
                s3_id = f"S3-{s3_id_counter:06d}"
                s3_id_counter += 1
                name2 = noisy_name(e["name"], 2)
                addr2 = noisy_address(e["num"], e["street"], e["street_type"], e["city"], e["state"], 2)
                s3_rows.append({"id": s3_id, "name": name2, "address": addr2})
                matched_s3.append(s3_id)

        gt_rows.append({
            "id": s1_id,
            "matches": ",".join(matched_s2 + matched_s3)
        })

    # Add a handful of pure-noise decoy records to source2/3 that don't
    # correspond to any source1 entity (realistic clutter / distractors).
    for _ in range(max(5, n // 20)):
        city, state = random.choice(CITIES)
        num, street, city, state = rand_address(city, state)
        street_type = random.choice(STREET_TYPES)
        name = rand_business_name()
        if random.random() < 0.5:
            s2_id = f"S2-{s2_id_counter:06d}"
            s2_id_counter += 1
            s2_rows.append({"id": s2_id, "name": noisy_name(name, 1),
                             "address": noisy_address(num, street, street_type, city, state, 1)})
        else:
            s3_id = f"S3-{s3_id_counter:06d}"
            s3_id_counter += 1
            s3_rows.append({"id": s3_id, "name": noisy_name(name, 2),
                             "address": noisy_address(num, street, street_type, city, state, 2)})

    random.shuffle(s1_rows)
    random.shuffle(s2_rows)
    random.shuffle(s3_rows)

    write_tsv(os.path.join(OUT_DIR, f"{split_name}_source1.tsv"), s1_rows, ["id", "name", "address"])
    write_tsv(os.path.join(OUT_DIR, f"{split_name}_source2.tsv"), s2_rows, ["id", "name", "address"])
    write_tsv(os.path.join(OUT_DIR, f"{split_name}_source3.tsv"), s3_rows, ["id", "name", "address"])
    if split_name == "train":
        write_tsv(os.path.join(OUT_DIR, "train_ground_truth.tsv"), gt_rows, ["id", "matches"])
    else:
        # NOTE: real challenges never give you this file -- it's written
        # here only so this demo pipeline can self-check its accuracy.
        write_tsv(os.path.join(OUT_DIR, "EVAL_ONLY_test_ground_truth.tsv"), gt_rows, ["id", "matches"])

    return s1_rows, s2_rows, s3_rows, gt_rows


if __name__ == "__main__":
    all_entities = build_entities(n=600)
    random.shuffle(all_entities)
    train_entities = all_entities[:450]
    test_entities = all_entities[450:]

    generate_split(train_entities, "train")
    generate_split(test_entities, "test")

    print("Synthetic data written to", os.path.abspath(OUT_DIR))
