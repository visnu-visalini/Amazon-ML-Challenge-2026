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

    return f"{country}|{name[:length]}"


def address_block(address, country, length):

    address = block_normalize(address)
    country = block_normalize(country)

    if not address:
        return None

    return f"{country}|{address[:length]}"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PRODUCTION BLOCKING RECALL VALIDATION")
print("=" * 70)


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
# LOAD S2 POSITIVE PAIRS
# ============================================================

print("\nLoading S2 training pairs...")

pairs = pd.read_csv(
    PAIR_FILE,
    dtype=str,
    keep_default_na=False
)

# Only TRUE matches are needed.
positives = pairs[
    pairs["label"] == "1"
].copy()

print(
    f"Total training pairs : {len(pairs):,}"
)

print(
    f"Positive pairs        : {len(positives):,}"
)


# ============================================================
# LOAD ONLY S2 RECORDS NEEDED
# ============================================================

print("\nLoading S2...")

needed_s2_ids = set(
    positives["source_entity_id"]
)

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s2 = s2[
    s2["entity_id"].isin(
        needed_s2_ids
    )
].copy()

print(
    f"S2 records needed: {len(s2):,}"
)


# ============================================================
# CREATE LOOKUPS
# ============================================================

print("\nCreating record lookups...")

s1_lookup = s1.set_index(
    "entity_id"
)[
    [
        "business_name",
        "business_address",
        "country"
    ]
].to_dict("index")

s2_lookup = s2.set_index(
    "entity_id"
)[
    [
        "business_name",
        "business_address",
        "country"
    ]
].to_dict("index")


# ============================================================
# BUILD S1 BLOCK INDEXES
# ============================================================

print("\nBuilding S1 indexes...")

indexes = {

    "address12": {},

    "address16": {},

    "name3_address12": {},

    "name3_address16": {}
}


def add(index, key, entity_id):

    if key is None:
        return

    index.setdefault(
        key,
        set()
    ).add(entity_id)


for entity_id, record in s1_lookup.items():

    name = record["business_name"]
    address = record["business_address"]
    country = record["country"]

    # --------------------------------------------------------
    # ADDRESS 12
    # --------------------------------------------------------

    add(
        indexes["address12"],
        address_block(
            address,
            country,
            12
        ),
        entity_id
    )

    # --------------------------------------------------------
    # ADDRESS 16
    # --------------------------------------------------------

    add(
        indexes["address16"],
        address_block(
            address,
            country,
            16
        ),
        entity_id
    )

    # --------------------------------------------------------
    # NAME 3 + ADDRESS 12
    # --------------------------------------------------------

    n3 = name_block(
        name,
        country,
        3
    )

    a12 = address_block(
        address,
        country,
        12
    )

    if n3 is not None and a12 is not None:

        add(
            indexes["name3_address12"],
            f"{n3}|{a12}",
            entity_id
        )

    # --------------------------------------------------------
    # NAME 3 + ADDRESS 16
    # --------------------------------------------------------

    a16 = address_block(
        address,
        country,
        16
    )

    if n3 is not None and a16 is not None:

        add(
            indexes["name3_address16"],
            f"{n3}|{a16}",
            entity_id
        )


for name, index in indexes.items():

    print(
        f"{name:20s}: "
        f"{len(index):,} blocks"
    )


# ============================================================
# VALIDATE EACH TRUE MATCH
# ============================================================

print("\n" + "=" * 70)
print("CHECKING TRUE MATCH RETENTION")
print("=" * 70)


strategies = [
    "address12",
    "address16",
    "name3_address12",
    "name3_address16"
]


preserved = {
    strategy: 0
    for strategy in strategies
}


total_positive = len(positives)


# ------------------------------------------------------------
# Each positive pair is:
#
# source1_entity_id
# source_entity_id
#
# We check whether the TRUE S1 ID appears
# inside the S2 record's block.
# ------------------------------------------------------------

for row_number, row in enumerate(
    positives.itertuples(index=False),
    start=1
):

    s1_id = row.source1_entity_id
    s2_id = row.source_entity_id

    s1_record = s1_lookup.get(
        s1_id
    )

    s2_record = s2_lookup.get(
        s2_id
    )

    if (
        s1_record is None
        or s2_record is None
    ):
        continue

    name = s2_record["business_name"]
    address = s2_record["business_address"]
    country = s2_record["country"]

    # --------------------------------------------------------
    # ADDRESS 12
    # --------------------------------------------------------

    candidates = indexes[
        "address12"
    ].get(
        address_block(
            address,
            country,
            12
        ),
        set()
    )

    if s1_id in candidates:

        preserved[
            "address12"
        ] += 1

    # --------------------------------------------------------
    # ADDRESS 16
    # --------------------------------------------------------

    candidates = indexes[
        "address16"
    ].get(
        address_block(
            address,
            country,
            16
        ),
        set()
    )

    if s1_id in candidates:

        preserved[
            "address16"
        ] += 1

    # --------------------------------------------------------
    # NAME 3 + ADDRESS 12
    # --------------------------------------------------------

    n3 = name_block(
        name,
        country,
        3
    )

    a12 = address_block(
        address,
        country,
        12
    )

    if n3 is not None and a12 is not None:

        candidates = indexes[
            "name3_address12"
        ].get(
            f"{n3}|{a12}",
            set()
        )

        if s1_id in candidates:

            preserved[
                "name3_address12"
            ] += 1

    # --------------------------------------------------------
    # NAME 3 + ADDRESS 16
    # --------------------------------------------------------

    a16 = address_block(
        address,
        country,
        16
    )

    if n3 is not None and a16 is not None:

        candidates = indexes[
            "name3_address16"
        ].get(
            f"{n3}|{a16}",
            set()
        )

        if s1_id in candidates:

            preserved[
                "name3_address16"
            ] += 1

    if row_number % 25_000 == 0:

        print(
            f"Checked positive pairs: "
            f"{row_number:,}/"
            f"{total_positive:,}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RECALL RESULTS")
print("=" * 70)

print(
    f"\nTotal positive S2 matches: "
    f"{total_positive:,}"
)

print()

for strategy in strategies:

    recall = (
        preserved[strategy]
        / total_positive
        * 100
    )

    print(
        f"{strategy:20s} | "
        f"preserved: "
        f"{preserved[strategy]:>8,} | "
        f"recall: "
        f"{recall:>9.4f}%"
    )


print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nDo NOT start full prediction yet."
)