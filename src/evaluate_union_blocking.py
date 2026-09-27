import pandas as pd
from pathlib import Path
import re
import unicodedata
from collections import defaultdict
import gc


# ============================================================
# PATHS
# ============================================================

BASE = Path("D:/Amazon ML Challenge/dataset/train")

S1_PATH = BASE / "train_source1.tsv"
S2_PATH = BASE / "train_source2.tsv"
S3_PATH = BASE / "train_source3.tsv"
GT_PATH = BASE / "train_ground_truth.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_name(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKC",
        value
    )

    value = value.lower()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def normalize_address(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKC",
        value
    )

    value = value.lower()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# BLOCKING KEYS
# ============================================================

def name_block(
    name,
    country
):

    name = normalize_name(name)

    country = str(
        country
    ).strip().lower()

    if not name:

        name_prefix = "#"

    else:

        # 2-character name prefix
        name_prefix = name[:2]

    return country + "|" + name_prefix


def address_block(
    address,
    country
):

    address = normalize_address(address)

    country = str(
        country
    ).strip().lower()

    if not address:

        return None

    # First 12 normalized address characters
    address_prefix = address[:12]

    return country + "|" + address_prefix


# ============================================================
# LOAD S1
# ============================================================

print("Loading S1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    dtype=str
)

print(
    f"S1 rows: {len(s1):,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
)


# ============================================================
# BUILD GROUND-TRUTH LOOKUP
# ============================================================

true_matches = defaultdict(set)

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id

    matched_ids = str(
        row.matched_entity_ids
    )

    if (
        not matched_ids
        or matched_ids.lower() == "nan"
    ):
        continue

    # Handle possible separators
    ids = re.split(
        r"[,;|]",
        matched_ids
    )

    for entity_id in ids:

        entity_id = entity_id.strip()

        if entity_id:
            true_matches[s1_id].add(
                entity_id
            )


del gt

gc.collect()


# ============================================================
# BUILD INDEX
# ============================================================

def build_indexes(
    source_path,
    source_name
):

    print("\n" + "=" * 60)
    print(
        f"BUILDING {source_name} INDEXES"
    )
    print("=" * 60)

    name_index = defaultdict(set)
    address_index = defaultdict(set)

    rows_processed = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        dtype=str,
        chunksize=200000
    ):

        for row in chunk.itertuples(
            index=False
        ):

            entity_id = row.entity_id

            # -----------------------------
            # NAME INDEX
            # -----------------------------

            nb = name_block(
                row.business_name,
                row.country
            )

            name_index[nb].add(
                entity_id
            )

            # -----------------------------
            # ADDRESS INDEX
            # -----------------------------

            ab = address_block(
                row.business_address,
                row.country
            )

            if ab is not None:

                address_index[ab].add(
                    entity_id
                )

        rows_processed += len(chunk)

        print(
            f"{source_name} rows indexed: "
            f"{rows_processed:,}"
        )

    print(
        f"{source_name} name blocks: "
        f"{len(name_index):,}"
    )

    print(
        f"{source_name} address blocks: "
        f"{len(address_index):,}"
    )

    return name_index, address_index


# ============================================================
# EVALUATE UNION BLOCKING
# ============================================================

def evaluate(
    name_index,
    address_index,
    source_name
):

    print("\n" + "=" * 60)
    print(
        f"EVALUATING UNION - {source_name}"
    )
    print("=" * 60)

    total_true = 0
    preserved_true = 0

    entities_with_matches = 0
    candidate_total = 0

    prefix = source_name + "-"

    print(
        "Starting S1 evaluation..."
    )

    for i, row in enumerate(
        s1.itertuples(index=False),
        start=1
    ):

        s1_id = row.entity_id

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        matches = true_matches.get(
            s1_id,
            ()
        )

        source_matches = []

        for match_id in matches:

            if match_id.startswith(prefix):

                source_matches.append(
                    match_id
                )

        if not source_matches:
            continue

        entities_with_matches += 1

        total_true += len(
            source_matches
        )

        # ----------------------------------------------------
        # NAME BLOCK
        # ----------------------------------------------------

        nb = name_block(
            row.business_name,
            row.country
        )

        name_candidates = name_index.get(
            nb
        )

        if name_candidates is None:

            name_candidates = ()

        # ----------------------------------------------------
        # ADDRESS BLOCK
        # ----------------------------------------------------

        ab = address_block(
            row.business_address,
            row.country
        )

        if ab is not None:

            address_candidates = address_index.get(
                ab
            )

            if address_candidates is None:

                address_candidates = ()

        else:

            address_candidates = ()

        # ----------------------------------------------------
        # CANDIDATE COUNT
        # ----------------------------------------------------

        # Exact union size without constructing
        # a new large set.
        #
        # Usually the overlap between the two blocks
        # is small, so this remains manageable.

        if (
            name_candidates
            and address_candidates
        ):

            overlap = len(
                name_candidates
                & address_candidates
            )

            candidate_total += (
                len(name_candidates)
                + len(address_candidates)
                - overlap
            )

        else:

            candidate_total += (
                len(name_candidates)
                + len(address_candidates)
            )

        # ----------------------------------------------------
        # RECALL
        # ----------------------------------------------------

        for match_id in source_matches:

            if (
                match_id in name_candidates
                or match_id in address_candidates
            ):

                preserved_true += 1

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if i % 250000 == 0:

            print(
                f"Processed S1 rows: {i:,} | "
                f"Entities with matches: "
                f"{entities_with_matches:,} | "
                f"Preserved links: "
                f"{preserved_true:,}"
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    recall = (
        preserved_true / total_true
        if total_true
        else 0
    )

    avg_candidates = (
        candidate_total / entities_with_matches
        if entities_with_matches
        else 0
    )

    print("\n" + "-" * 60)

    print(
        f"Entities with matches: "
        f"{entities_with_matches:,}"
    )

    print(
        f"True links: "
        f"{total_true:,}"
    )

    print(
        f"True links preserved: "
        f"{preserved_true:,}"
    )

    print(
        f"Blocking recall: "
        f"{recall * 100:.4f}%"
    )

    print(
        f"Average candidates/entity: "
        f"{avg_candidates:,.2f}"
    )

    print("-" * 60)

    return {
        "source": source_name,
        "entities_with_matches": entities_with_matches,
        "true_links": total_true,
        "preserved_true": preserved_true,
        "recall": recall,
        "avg_candidates": avg_candidates
    }


# ============================================================
# S2
# ============================================================

s2_name, s2_address = build_indexes(
    S2_PATH,
    "S2"
)

s2_result = evaluate(
    s2_name,
    s2_address,
    "S2"
)


# ============================================================
# FREE S2 MEMORY
# ============================================================

print("\nFreeing S2 indexes...")

del s2_name
del s2_address

gc.collect()

print("S2 memory released.")


# ============================================================
# S3
# ============================================================

s3_name, s3_address = build_indexes(
    S3_PATH,
    "S3"
)

s3_result = evaluate(
    s3_name,
    s3_address,
    "S3"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("UNION BLOCKING EVALUATION COMPLETE")
print("=" * 60)

print(
    f"\nS2 recall: "
    f"{s2_result['recall'] * 100:.4f}%"
)

print(
    f"S2 average candidates/entity: "
    f"{s2_result['avg_candidates']:,.2f}"
)

print(
    f"\nS3 recall: "
    f"{s3_result['recall'] * 100:.4f}%"
)

print(
    f"S3 average candidates/entity: "
    f"{s3_result['avg_candidates']:,.2f}"
)

print("\nDone.")