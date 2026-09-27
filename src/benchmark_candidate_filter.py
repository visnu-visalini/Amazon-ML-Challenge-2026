import re
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"D:\Amazon ML Challenge")

TRAIN_DIR = BASE_DIR / "dataset" / "train"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"

PAIR_FILE = BASE_DIR / "training_data" / "s2_training_pairs.csv"

SAMPLE_SIZE = 100_000
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


def token_set(value):

    return set(
        normalize(value).split()
    )


def jaccard(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    return intersection / union if union else 0.0


def character_similarity(a, b):

    a = normalize(a)
    b = normalize(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    shorter = min(len(a), len(b))

    if shorter == 0:
        return 0.0

    # Cheap character overlap using character sets.
    # This is intentionally much cheaper than the full
    # LightGBM feature calculation.

    ca = set(a)
    cb = set(b)

    return len(ca & cb) / len(ca | cb)


# ============================================================
# BLOCK KEY
# ============================================================

def block_normalize(value):

    return normalize(value)


def make_name2(name, country):

    name = block_normalize(name)
    country = block_normalize(country)

    if not name:
        return None

    return f"{country}|{name[:2]}"


def make_address12(address, country):

    address = block_normalize(address)
    country = block_normalize(country)

    if not address:
        return None

    return f"{country}|{address[:12]}"


# ============================================================
# LOAD TRAINING POSITIVES
# ============================================================

print("=" * 70)
print("CHEAP CANDIDATE FILTER BENCHMARK")
print("=" * 70)

pairs = pd.read_csv(
    PAIR_FILE,
    dtype=str,
    keep_default_na=False
)

positives = pairs[
    pairs["label"] == "1"
].copy()

print(
    f"\nPositive pairs available: "
    f"{len(positives):,}"
)


# ============================================================
# LOAD S1
# ============================================================

print("\nLoading S1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(
    f"S1 rows: {len(s1):,}"
)


# ============================================================
# BUILD BLOCK INDEX
# ============================================================

print("\nBuilding production block indexes...")

name2_index = {}
address12_index = {}

for row in s1.itertuples(index=False):

    name_key = make_name2(
        row.business_name,
        row.country
    )

    address_key = make_address12(
        row.business_address,
        row.country
    )

    if name_key:

        name2_index.setdefault(
            name_key,
            []
        ).append(row.entity_id)

    if address_key:

        address12_index.setdefault(
            address_key,
            []
        ).append(row.entity_id)


print(
    f"name2 blocks    : {len(name2_index):,}"
)

print(
    f"address12 blocks : {len(address12_index):,}"
)

s1_lookup = (
    s1.set_index("entity_id")
    [
        [
            "business_name",
            "business_address",
            "country"
        ]
    ]
    .to_dict("index")
)


# ============================================================
# LOAD RANDOM-ISH S2 SAMPLE
# ============================================================

print(
    f"\nLoading first {SAMPLE_SIZE:,} S2 rows..."
)

s2_sample = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=SAMPLE_SIZE
)

print(
    f"S2 sample: {len(s2_sample):,}"
)


# ============================================================
# CANDIDATE FILTERS
# ============================================================

FILTERS = {

    # Very permissive
    "name_jaccard_0.20": 0.20,

    # Moderate
    "name_jaccard_0.30": 0.30,

    # More selective
    "name_jaccard_0.40": 0.40,

    # Strict
    "name_jaccard_0.50": 0.50,
}


candidate_counts = {
    name: 0
    for name in FILTERS
}

block_candidate_count = 0

processed = 0


# ============================================================
# PROCESS SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING SAMPLE")
print("=" * 70)


for row in s2_sample.itertuples(index=False):

    name_key = make_name2(
        row.business_name,
        row.country
    )

    address_key = make_address12(
        row.business_address,
        row.country
    )

    candidate_ids = set()

    # --------------------------------------------------------
    # name2 block
    # --------------------------------------------------------

    if name_key in name2_index:

        candidate_ids.update(
            name2_index[name_key]
        )

    # --------------------------------------------------------
    # address12 block
    # --------------------------------------------------------

    if address_key in address12_index:

        candidate_ids.update(
            address12_index[address_key]
        )

    block_candidate_count += len(
        candidate_ids
    )

    # --------------------------------------------------------
    # Cheap filter
    # --------------------------------------------------------

    s2_name_tokens = token_set(
        row.business_name
    )

    for candidate_id in candidate_ids:

        # Find candidate record.
        # S1 is indexed by entity_id below.
        candidate = s1_lookup[candidate_id]

        candidate_tokens = token_set(
            candidate["business_name"]
        )

        score = jaccard(
            s2_name_tokens,
            candidate_tokens
        )

        for filter_name, threshold in FILTERS.items():

            if score >= threshold:

                candidate_counts[
                    filter_name
                ] += 1

    processed += 1

    if processed % 10_000 == 0:

        print(
            f"Processed: "
            f"{processed:,}/{len(s2_sample):,}"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE REDUCTION RESULTS")
print("=" * 70)

average_block_candidates = (
    block_candidate_count
    / processed
)

print(
    f"\nOriginal blocked candidates:"
    f" {block_candidate_count:,}"
)

print(
    f"Average candidates/S2:"
    f" {average_block_candidates:.2f}"
)

print()

for name, count in candidate_counts.items():

    average = count / processed

    reduction = (
        1
        - count / block_candidate_count
    ) * 100

    print(
        f"{name:25s} | "
        f"total: {count:12,} | "
        f"avg/S2: {average:8.2f} | "
        f"reduction: {reduction:7.2f}%"
    )


print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

print(
    "\nImportant: this benchmark measures "
    "candidate reduction only."
)

print(
    "We have NOT yet verified recall of "
    "these cheap filters."
)

print(
    "\nDo NOT start full prediction yet."
)