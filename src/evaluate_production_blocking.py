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
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

SAMPLE_ROWS = 100_000


# ============================================================
# NORMALIZATION
# ============================================================

def block_normalize(value):

    if value is None:
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKC",
        value
    )

    value = value.lower()

    value = re.sub(
        r"\W+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# BLOCK KEYS
# ============================================================

def name_block(name, country, length):

    name = block_normalize(name)
    country = block_normalize(country)

    if not name:
        return None

    return (
        f"{country}|"
        f"{name[:length]}"
    )


def address_block(address, country, length):

    address = block_normalize(address)
    country = block_normalize(country)

    if not address:
        return None

    return (
        f"{country}|"
        f"{address[:length]}"
    )


# ============================================================
# LOAD S1
# ============================================================

print("=" * 70)
print("PRODUCTION BLOCKING BENCHMARK")
print("=" * 70)

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
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(
    f"Ground-truth rows: {len(gt):,}"
)


# ============================================================
# BUILD TRUE MATCH LOOKUP
# ============================================================

print("\nBuilding ground-truth lookup...")

true_matches = {}

for row in gt.itertuples(index=False):

    matched = str(
        row.matched_entity_ids
    ).strip()

    if not matched:
        true_matches[
            row.source1_entity_id
        ] = set()

        continue

    # Handle common separators safely.
    ids = re.split(
        r"[,;| ]+",
        matched
    )

    true_matches[
        row.source1_entity_id
    ] = set(
        x for x in ids if x
    )


# ============================================================
# BUILD S1 INDEXES
# ============================================================

print("\nBuilding S1 block indexes...")

strategies = {

    "name2": {},
    "name3": {},

    "address12": {},
    "address16": {},

    "name3_address12": {},
    "name3_address16": {}
}


def add_to_index(index, key, s1_id):

    if key is None:
        return

    index.setdefault(
        key,
        set()
    ).add(s1_id)


for row in s1.itertuples(index=False):

    s1_id = row.entity_id
    country = row.country

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    add_to_index(
        strategies["name2"],
        name_block(
            row.business_name,
            country,
            2
        ),
        s1_id
    )

    add_to_index(
        strategies["name3"],
        name_block(
            row.business_name,
            country,
            3
        ),
        s1_id
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    add_to_index(
        strategies["address12"],
        address_block(
            row.business_address,
            country,
            12
        ),
        s1_id
    )

    add_to_index(
        strategies["address16"],
        address_block(
            row.business_address,
            country,
            16
        ),
        s1_id
    )

    # --------------------------------------------------------
    # COMBINED
    # --------------------------------------------------------

    n3 = name_block(
        row.business_name,
        country,
        3
    )

    a12 = address_block(
        row.business_address,
        country,
        12
    )

    a16 = address_block(
        row.business_address,
        country,
        16
    )

    if n3 is not None and a12 is not None:

        add_to_index(
            strategies["name3_address12"],
            f"{n3}|{a12}",
            s1_id
        )

    if n3 is not None and a16 is not None:

        add_to_index(
            strategies["name3_address16"],
            f"{n3}|{a16}",
            s1_id
        )


for name, index in strategies.items():

    print(
        f"{name:20s}: "
        f"{len(index):,} blocks"
    )


# ============================================================
# LOAD S2 SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("LOADING S2 SAMPLE")
print("=" * 70)

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=SAMPLE_ROWS
)

print(
    f"S2 sample rows: {len(s2):,}"
)


# ============================================================
# CREATE S2 → S1 CANDIDATES
# ============================================================

results = {

    name: {
        "candidates": 0,
        "true_found": 0,
        "true_total": 0
    }

    for name in strategies
}


# ------------------------------------------------------------
# IMPORTANT:
# We evaluate each S2 record against S1.
# This measures how many S1 candidates each S2 record creates.
# ------------------------------------------------------------

for row_number, row in enumerate(
    s2.itertuples(index=False),
    start=1
):

    country = row.country

    true_s1_ids = set()

    # Ground truth is indexed by S1, so we don't know
    # the true S1 directly from an S2 row.
    #
    # Therefore candidate volume is measured here.
    # Recall will be checked separately using known
    # positive training pairs.

    candidate_sets = {

        "name2": set(),

        "name3": set(),

        "address12": set(),

        "address16": set(),

        "name3_address12": set(),

        "name3_address16": set()
    }

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    candidate_sets["name2"] = strategies[
        "name2"
    ].get(
        name_block(
            row.business_name,
            country,
            2
        ),
        set()
    )

    candidate_sets["name3"] = strategies[
        "name3"
    ].get(
        name_block(
            row.business_name,
            country,
            3
        ),
        set()
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    candidate_sets["address12"] = strategies[
        "address12"
    ].get(
        address_block(
            row.business_address,
            country,
            12
        ),
        set()
    )

    candidate_sets["address16"] = strategies[
        "address16"
    ].get(
        address_block(
            row.business_address,
            country,
            16
        ),
        set()
    )

    # --------------------------------------------------------
    # COMBINED
    # --------------------------------------------------------

    n3 = name_block(
        row.business_name,
        country,
        3
    )

    a12 = address_block(
        row.business_address,
        country,
        12
    )

    a16 = address_block(
        row.business_address,
        country,
        16
    )

    if n3 is not None and a12 is not None:

        candidate_sets[
            "name3_address12"
        ] = strategies[
            "name3_address12"
        ].get(
            f"{n3}|{a12}",
            set()
        )

    if n3 is not None and a16 is not None:

        candidate_sets[
            "name3_address16"
        ] = strategies[
            "name3_address16"
        ].get(
            f"{n3}|{a16}",
            set()
        )

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    for strategy, candidates in candidate_sets.items():

        results[strategy][
            "candidates"
        ] += len(candidates)

    if row_number % 10_000 == 0:

        print(
            f"Processed S2 rows: "
            f"{row_number:,}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BLOCKING CANDIDATE VOLUME")
print("=" * 70)

print(
    f"\nSample size: {len(s2):,}"
)

print()

for strategy, result in results.items():

    average = (
        result["candidates"]
        / len(s2)
    )

    print(
        f"{strategy:20s} | "
        f"total candidates: "
        f"{result['candidates']:>12,} | "
        f"average/S2: "
        f"{average:>10.2f}"
    )


print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

print(
    "\nThis test measures candidate volume only."
)

print(
    "Do NOT start full prediction yet."
)