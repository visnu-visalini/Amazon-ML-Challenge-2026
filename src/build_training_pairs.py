import os
import gc
import random
import pandas as pd

from normalization import normalize_business_name, normalize_address


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"D:\Amazon ML Challenge"
DATA_DIR = os.path.join(BASE_DIR, "dataset", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "training_data")

S1_FILE = os.path.join(DATA_DIR, "train_source1.tsv")
GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")

SOURCE_CONFIG = {
    "S2": os.path.join(DATA_DIR, "train_source2.tsv"),
    "S3": os.path.join(DATA_DIR, "train_source3.tsv"),
}

CHUNK_SIZE = 100_000

# Instead of scanning all 2.2M S1 rows,
# use a representative sample.
SAMPLE_S1 = 300_000

# Maximum pairs per source
MAX_POSITIVES = 100_000
MAX_NEGATIVES = 150_000

# Hard negatives per S1 entity
NEGATIVES_PER_ENTITY = 2

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


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
# LOAD GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("LOADING GROUND TRUTH")
print("=" * 70)

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"Ground truth rows: {len(gt):,}")


# ============================================================
# GROUND TRUTH LOOKUP
# ============================================================

print("\nBuilding ground-truth lookup...")

gt_lookup = {}

for row in gt.itertuples(index=False):

    source1_id = row.source1_entity_id
    matched = row.matched_entity_ids

    if matched:
        gt_lookup[source1_id] = set(
            x.strip()
            for x in matched.split(",")
            if x.strip()
        )
    else:
        gt_lookup[source1_id] = set()

print(f"GT lookup entries: {len(gt_lookup):,}")


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

s1["name_prefix"] = s1["name_norm"].str[:2]
s1["address_prefix"] = s1["address_norm"].str[:12]

print("S1 normalization complete.")


# ============================================================
# SAMPLE S1
# ============================================================

print("\nSelecting S1 sample...")

# Prefer entities having at least one known match.
matched_s1 = [
    entity_id
    for entity_id in s1["entity_id"]
    if gt_lookup.get(entity_id)
]

unmatched_s1 = [
    entity_id
    for entity_id in s1["entity_id"]
    if not gt_lookup.get(entity_id)
]

random.shuffle(matched_s1)
random.shuffle(unmatched_s1)

matched_sample_size = int(SAMPLE_S1 * 0.8)
unmatched_sample_size = SAMPLE_S1 - matched_sample_size

selected_ids = (
    matched_s1[:matched_sample_size]
    + unmatched_s1[:unmatched_sample_size]
)

selected_ids = set(selected_ids)

print(f"Selected S1 entities: {len(selected_ids):,}")
print(f"  Matched entities: {min(matched_sample_size, len(matched_s1)):,}")
print(f"  Unmatched entities: {min(unmatched_sample_size, len(unmatched_s1)):,}")


# Keep only selected S1 rows
s1_sample = s1[
    s1["entity_id"].isin(selected_ids)
].copy()

print(f"S1 sample rows: {len(s1_sample):,}")


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(source_name, source_file):

    print("\n" + "=" * 70)
    print(f"PROCESSING {source_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # BUILD INDEXES
    # --------------------------------------------------------

    name_index = {}
    address_index = {}

    print(f"\nBuilding {source_name} indexes...")

    for chunk in pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        chunk["name_norm"] = chunk["business_name"].map(
            normalize_business_name
        )

        chunk["address_norm"] = chunk["business_address"].map(
            normalize_address
        )

        chunk["name_prefix"] = chunk["name_norm"].str[:2]
        chunk["address_prefix"] = chunk["address_norm"].str[:12]

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            name_key = (
                row.country,
                row.name_prefix
            )

            address_key = (
                row.country,
                row.address_prefix
            )

            name_index.setdefault(
                name_key, []
            ).append(entity_id)

            address_index.setdefault(
                address_key, []
            ).append(entity_id)

        del chunk
        gc.collect()

    print(f"Name blocks: {len(name_index):,}")
    print(f"Address blocks: {len(address_index):,}")

    # --------------------------------------------------------
    # GENERATE PAIRS
    # --------------------------------------------------------

    positives = []
    negatives = []

    positive_count = 0
    negative_count = 0

    processed = 0

    print("\nGenerating training pairs...")

    for row in s1_sample.itertuples(index=False):

        s1_id = row.entity_id

        true_matches = gt_lookup.get(
            s1_id,
            set()
        )

        # ----------------------------------------------------
        # GET CANDIDATES
        # ----------------------------------------------------

        name_key = (
            row.country,
            row.name_prefix
        )

        address_key = (
            row.country,
            row.address_prefix
        )

        name_candidates = name_index.get(
            name_key,
            []
        )

        address_candidates = address_index.get(
            address_key,
            []
        )

        # ----------------------------------------------------
        # POSITIVES
        # ----------------------------------------------------

        if true_matches:

            for candidate_id in true_matches:

                # Only retain positives captured
                # by our blocking strategy.
                if (
                    candidate_id in name_candidates
                    or
                    candidate_id in address_candidates
                ):

                    positives.append({
                        "source1_entity_id": s1_id,
                        "source_entity_id": candidate_id,
                        "source": source_name,
                        "label": 1
                    })

                    positive_count += 1

                    if positive_count >= MAX_POSITIVES:
                        break

        # ----------------------------------------------------
        # HARD NEGATIVES
        # ----------------------------------------------------

        if negative_count < MAX_NEGATIVES:

            # Sample from the smaller candidate list first.
            #
            # This avoids constructing huge unions.

            if (
                name_candidates
                and address_candidates
            ):

                if len(name_candidates) <= len(address_candidates):
                    candidate_pool = name_candidates
                else:
                    candidate_pool = address_candidates

            elif name_candidates:

                candidate_pool = name_candidates

            else:

                candidate_pool = address_candidates

            if candidate_pool:

                # We only inspect a small number of random
                # positions instead of shuffling thousands
                # of candidates.

                pool_size = len(candidate_pool)

                attempts = min(
                    NEGATIVES_PER_ENTITY * 5,
                    pool_size
                )

                sampled_indices = random.sample(
                    range(pool_size),
                    attempts
                )

                selected = []

                for idx in sampled_indices:

                    candidate_id = candidate_pool[idx]

                    if candidate_id not in true_matches:

                        selected.append(candidate_id)

                        if len(selected) >= NEGATIVES_PER_ENTITY:
                            break

                for candidate_id in selected:

                    negatives.append({
                        "source1_entity_id": s1_id,
                        "source_entity_id": candidate_id,
                        "source": source_name,
                        "label": 0
                    })

                    negative_count += 1

                    if negative_count >= MAX_NEGATIVES:
                        break

        processed += 1

        if processed % 50_000 == 0:

            print(
                f"Processed: {processed:,} | "
                f"Positives: {positive_count:,} | "
                f"Negatives: {negative_count:,}"
            )

        if (
            positive_count >= MAX_POSITIVES
            and
            negative_count >= MAX_NEGATIVES
        ):
            break

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    result = pd.DataFrame(
        positives + negatives
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{source_name.lower()}_training_pairs.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print("\n" + "-" * 70)
    print(f"{source_name} COMPLETE")
    print("-" * 70)

    print(f"Pairs saved: {len(result):,}")
    print(
        f"Positive pairs: "
        f"{(result['label'] == 1).sum():,}"
    )
    print(
        f"Negative pairs: "
        f"{(result['label'] == 0).sum():,}"
    )

    print(f"Output: {output_file}")

    del name_index
    del address_index
    del result
    del positives
    del negatives

    gc.collect()


# ============================================================
# RUN S2
# ============================================================

process_source(
    "S2",
    SOURCE_CONFIG["S2"]
)


# ============================================================
# RUN S3
# ============================================================

process_source(
    "S3",
    SOURCE_CONFIG["S3"]
)


print("\n" + "=" * 70)
print("TRAINING PAIR GENERATION FINISHED")
print("=" * 70)