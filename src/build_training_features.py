import os
import gc
import pandas as pd

from normalization import (
    normalize_business_name,
    normalize_address
)

from similarity_features import calculate_features


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"D:\Amazon ML Challenge"
DATA_DIR = os.path.join(BASE_DIR, "dataset", "train")
PAIR_DIR = os.path.join(BASE_DIR, "training_data")

S1_FILE = os.path.join(
    DATA_DIR,
    "train_source1.tsv"
)

SOURCE_FILES = {
    "S2": os.path.join(DATA_DIR, "train_source2.tsv"),
    "S3": os.path.join(DATA_DIR, "train_source3.tsv")
}

PAIR_FILES = {
    "S2": os.path.join(
        PAIR_DIR,
        "s2_training_pairs.csv"
    ),
    "S3": os.path.join(
        PAIR_DIR,
        "s3_training_pairs.csv"
    )
}

OUTPUT_FILES = {
    "S2": os.path.join(
        PAIR_DIR,
        "s2_training_features.csv"
    ),
    "S3": os.path.join(
        PAIR_DIR,
        "s3_training_features.csv"
    )
}


# ============================================================
# LOAD S1
# ============================================================

print("=" * 70)
print("LOADING S1")
print("=" * 70)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"S1 rows: {len(s1):,}")


# ============================================================
# NORMALIZE S1
# ============================================================

print("\nNormalizing S1...")

s1["name_norm"] = s1["business_name"].map(
    normalize_business_name
)

s1["address_norm"] = s1["business_address"].map(
    normalize_address
)


# ============================================================
# S1 LOOKUP
# ============================================================

s1_lookup = s1.set_index("entity_id")[
    [
        "business_name",
        "business_address",
        "country",
        "name_norm",
        "address_norm"
    ]
].to_dict("index")

print(
    f"S1 lookup entries: "
    f"{len(s1_lookup):,}"
)

del s1
gc.collect()


# ============================================================
# PROCESS SOURCE
# ============================================================

def process_source(source_name):

    print("\n" + "=" * 70)
    print(f"PROCESSING {source_name}")
    print("=" * 70)

    pair_file = PAIR_FILES[source_name]
    source_file = SOURCE_FILES[source_name]
    output_file = OUTPUT_FILES[source_name]

    # --------------------------------------------------------
    # LOAD PAIRS
    # --------------------------------------------------------

    print("\nLoading training pairs...")

    pairs = pd.read_csv(
        pair_file,
        dtype=str,
        keep_default_na=False
    )

    print(
        f"Training pairs: "
        f"{len(pairs):,}"
    )

    # --------------------------------------------------------
    # LOAD SOURCE
    # --------------------------------------------------------

    print(f"\nLoading {source_name}...")

    source = pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(
        f"{source_name} rows: "
        f"{len(source):,}"
    )

    # --------------------------------------------------------
    # SOURCE LOOKUP
    # --------------------------------------------------------

    source_lookup = source.set_index("entity_id")[
        [
            "business_name",
            "business_address",
            "country"
        ]
    ].to_dict("index")

    print(
        f"{source_name} lookup entries: "
        f"{len(source_lookup):,}"
    )

    del source
    gc.collect()

    # --------------------------------------------------------
    # FEATURE GENERATION
    # --------------------------------------------------------

    feature_rows = []

    total = len(pairs)

    print("\nGenerating similarity features...")

    for i, row in enumerate(
        pairs.itertuples(index=False),
        start=1
    ):

        s1_record = s1_lookup.get(
            row.source1_entity_id
        )

        source_record = source_lookup.get(
            row.source_entity_id
        )

        if (
            s1_record is None
            or source_record is None
        ):
            continue

        features = calculate_features(
            s1_record["business_name"],
            s1_record["business_address"],
            s1_record["country"],

            source_record["business_name"],
            source_record["business_address"],
            source_record["country"]
        )

        features["source1_entity_id"] = (
            row.source1_entity_id
        )

        features["source_entity_id"] = (
            row.source_entity_id
        )

        features["source"] = row.source

        features["label"] = int(row.label)

        feature_rows.append(features)

        if i % 25_000 == 0:

            print(
                f"Processed: "
                f"{i:,}/{total:,}"
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result = pd.DataFrame(feature_rows)

    print("\n" + "-" * 70)

    print(
        f"Final feature rows: "
        f"{len(result):,}"
    )

    print(
        f"Positive rows: "
        f"{(result['label'] == 1).sum():,}"
    )

    print(
        f"Negative rows: "
        f"{(result['label'] == 0).sum():,}"
    )

    print("\nFeature columns:")

    for column in result.columns:
        print(f"  {column}")

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved to:\n"
        f"{output_file}"
    )

    del pairs
    del source_lookup
    del feature_rows
    del result

    gc.collect()


# ============================================================
# S2
# ============================================================

process_source("S2")


# ============================================================
# S3
# ============================================================

process_source("S3")


print("\n" + "=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)