import re
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"D:\Amazon ML Challenge")

TRAIN_DIR = BASE_DIR / "dataset" / "train"
PAIR_FILE = BASE_DIR / "training_data" / "s2_training_pairs.csv"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"

CHUNK_SIZE = 250_000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value):

    if value is None:
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = value.lower()
    value = re.sub(r"\W+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ============================================================
# BLOCK KEYS
# ============================================================

def make_keys(record):

    name = normalize(record["business_name"])
    address = normalize(record["business_address"])
    country = normalize(record["country"])

    return {
        "name2": (
            f"{country}|{name[:2]}"
            if name else None
        ),

        "name3": (
            f"{country}|{name[:3]}"
            if name else None
        ),

        "address12": (
            f"{country}|{address[:12]}"
            if address else None
        ),

        "address16": (
            f"{country}|{address[:16]}"
            if address else None
        ),

        "name3_address12": (
            f"{country}|{name[:3]}|{address[:12]}"
            if name and address else None
        ),

        "name3_address16": (
            f"{country}|{name[:3]}|{address[:16]}"
            if name and address else None
        ),
    }


# ============================================================
# LOAD POSITIVE PAIRS
# ============================================================

print("=" * 70)
print("CORRECTED BLOCKING COMBINATION RECALL TEST")
print("=" * 70)

pairs = pd.read_csv(
    PAIR_FILE,
    dtype=str,
    keep_default_na=False
)

positives = pairs[
    pairs["label"] == "1"
][
    ["source1_entity_id", "source_entity_id"]
].copy()

print(
    f"\nPositive pairs: {len(positives):,}"
)


# ============================================================
# LOAD REQUIRED RECORDS
# ============================================================

def load_required_records(
    file_path,
    required_ids,
    source_name
):

    records = {}

    print(
        f"\nLoading required {source_name} records..."
    )

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        matched = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for row in matched.itertuples(
            index=False
        ):

            records[row.entity_id] = {
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country
            }

        if len(records) >= len(required_ids):
            break

    print(
        f"{source_name} records loaded: "
        f"{len(records):,}"
    )

    return records


s1_ids = set(
    positives["source1_entity_id"]
)

s2_ids = set(
    positives["source_entity_id"]
)

s1_records = load_required_records(
    S1_FILE,
    s1_ids,
    "S1"
)

s2_records = load_required_records(
    S2_FILE,
    s2_ids,
    "S2"
)


# ============================================================
# PRE-COMPUTE KEYS
# ============================================================

print("\nComputing keys...")

s1_keys = {
    entity_id: make_keys(record)
    for entity_id, record
    in s1_records.items()
}

s2_keys = {
    entity_id: make_keys(record)
    for entity_id, record
    in s2_records.items()
}


# ============================================================
# BLOCK MATCH FUNCTION
# ============================================================

def same_block(
    s1,
    s2,
    block_name
):

    return (
        s1[block_name] is not None
        and
        s1[block_name] == s2[block_name]
    )


# ============================================================
# CORRECT OR COMBINATIONS
#
# Each tuple means:
#
#     BLOCK A OR BLOCK B
#
# NOT:
#
#     BLOCK A AND BLOCK B
# ============================================================

strategies = {

    "address12": [
        "address12"
    ],

    "address16": [
        "address16"
    ],

    "name3_address12": [
        "name3_address12"
    ],

    "name3_address16": [
        "name3_address16"
    ],

    "address12 OR name3_address12": [
        "address12",
        "name3_address12"
    ],

    "address12 OR name3_address16": [
        "address12",
        "name3_address16"
    ],

    "address16 OR name3_address12": [
        "address16",
        "name3_address12"
    ],

    "address16 OR name3_address16": [
        "address16",
        "name3_address16"
    ],

    "address12 OR address16": [
        "address12",
        "address16"
    ],

    "address12 OR address16 OR name3_address12": [
        "address12",
        "address16",
        "name3_address12"
    ],

    "address12 OR address16 OR name3_address16": [
        "address12",
        "address16",
        "name3_address16"
    ],

    # Original production-style baseline
    "name2 OR address12": [
        "name2",
        "address12"
    ],
}


# ============================================================
# VALIDATE
# ============================================================

results = {
    name: 0
    for name in strategies
}

missing = 0
total = len(positives)

print("\n" + "=" * 70)
print("CHECKING TRUE MATCH RETENTION")
print("=" * 70)


for count, row in enumerate(
    positives.itertuples(index=False),
    start=1
):

    s1_id = row.source1_entity_id
    s2_id = row.source_entity_id

    if (
        s1_id not in s1_keys
        or s2_id not in s2_keys
    ):
        missing += 1
        continue

    k1 = s1_keys[s1_id]
    k2 = s2_keys[s2_id]

    for strategy_name, blocks in strategies.items():

        # IMPORTANT:
        # A match is preserved if ANY block matches.

        matched = any(
            same_block(
                k1,
                k2,
                block
            )
            for block in blocks
        )

        if matched:
            results[strategy_name] += 1

    if count % 25_000 == 0:

        print(
            f"Checked: "
            f"{count:,}/{total:,}"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED RECALL RESULTS")
print("=" * 70)

print(
    f"\nTotal positive pairs: {total:,}"
)

if missing:
    print(
        f"Missing records: {missing:,}"
    )

print()

for strategy, count in results.items():

    recall = (
        count / total * 100
    )

    print(
        f"{strategy:48s} | "
        f"{count:8,} | "
        f"{recall:8.4f}%"
    )


print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nDo NOT start full prediction yet."
) 