import pandas as pd
import re
from collections import defaultdict

BASE = r"D:\Amazon ML Challenge"
TRAIN = BASE + r"\dataset\train"

S1 = TRAIN + r"\train_source1.tsv"
S2 = TRAIN + r"\train_source2.tsv"
S3 = TRAIN + r"\train_source3.tsv"
GT = TRAIN + r"\train_ground_truth.tsv"


def norm(v):
    if not v:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(v).lower())


def nkey(country, name, n):
    x = norm(name)
    return f"{str(country).lower().strip()}|{x[:n]}" if x else None


def akey(country, address, n):
    x = norm(address)
    return f"{str(country).lower().strip()}|{x[:n]}" if x else None


print("Loading S1...")

s1 = pd.read_csv(
    S1, sep="\t", dtype=str, keep_default_na=False
)

print(f"S1: {len(s1):,}")

# Maps
s1_name3 = defaultdict(set)
s1_name4 = defaultdict(set)
s1_name5 = defaultdict(set)

s1_addr12 = defaultdict(set)
s1_addr16 = defaultdict(set)

for r in s1.itertuples(index=False):

    eid = r.entity_id
    c = r.country

    for d, n in [
        (s1_name3, 3),
        (s1_name4, 4),
        (s1_name5, 5),
    ]:
        k = nkey(c, r.business_name, n)
        if k:
            d[k].add(eid)

    for d, n in [
        (s1_addr12, 12),
        (s1_addr16, 16),
    ]:
        k = akey(c, r.business_address, n)
        if k:
            d[k].add(eid)

print("Indexes ready.")


def evaluate(source_file, label):

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    # source entity -> true S1 IDs
    print("Loading ground truth...")

    gt = pd.read_csv(
        GT,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    true_map = {}

    for r in gt.itertuples(index=False):

        ids = str(r.matched_entity_ids)

        if ids.strip():
            true_map[r.source1_entity_id] = set(
                x.strip() for x in ids.split(",") if x.strip()
            )

    # We need inverse GT: source entity -> S1
    # Ground truth is S1 -> source IDs.
    reverse_gt = defaultdict(set)

    for s1_id, source_ids in true_map.items():
        for sid in source_ids:
            reverse_gt[sid].add(s1_id)

    del gt
    del true_map

    print(f"GT source entities: {len(reverse_gt):,}")

    total_true = 0

    found = {
        "name3_addr16": 0,
        "name3_addr12": 0,
        "addr16": 0,
        "addr12": 0,
        "name4_addr16": 0,
        "name5_addr16": 0,
        "name4_addr12": 0,
        "name5_addr12": 0,
    }

    rows = 0

    for chunk in pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100000
    ):

        for r in chunk.itertuples(index=False):

            sid = r.entity_id
            truths = reverse_gt.get(sid)

            if not truths:
                continue

            rows += 1
            total_true += len(truths)

            c = r.country

            n3 = nkey(c, r.business_name, 3)
            n4 = nkey(c, r.business_name, 4)
            n5 = nkey(c, r.business_name, 5)

            a12 = akey(c, r.business_address, 12)
            a16 = akey(c, r.business_address, 16)

            blocks = {
                "name3_addr16":
                    (s1_name3.get(n3, set()) |
                     s1_addr16.get(a16, set())),

                "name3_addr12":
                    (s1_name3.get(n3, set()) |
                     s1_addr12.get(a12, set())),

                "addr16":
                    s1_addr16.get(a16, set()),

                "addr12":
                    s1_addr12.get(a12, set()),

                "name4_addr16":
                    (s1_name4.get(n4, set()) |
                     s1_addr16.get(a16, set())),

                "name5_addr16":
                    (s1_name5.get(n5, set()) |
                     s1_addr16.get(a16, set())),

                "name4_addr12":
                    (s1_name4.get(n4, set()) |
                     s1_addr12.get(a12, set())),

                "name5_addr12":
                    (s1_name5.get(n5, set()) |
                     s1_addr12.get(a12, set())),
            }

            for key, cand in blocks.items():
                found[key] += len(truths & cand)

        print(
            f"Processed {rows:,} true source entities",
            flush=True
        )

    print("\nRESULTS")
    print("-" * 70)

    for key, value in found.items():

        recall = (
            100 * value / total_true
            if total_true else 0
        )

        print(
            f"{key:18s} "
            f"{value:12,} / {total_true:,} "
            f"= {recall:8.3f}%"
        )


evaluate(S2, "S2")
evaluate(S3, "S3")