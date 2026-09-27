import os
import sys
from collections import defaultdict

import pandas as pd

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = r"D:\Amazon ML Challenge"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")

S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

CHUNK_SIZE = 100_000

# ------------------------------------------------------------
# Import existing normalization
# ------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization import normalize_address


# ------------------------------------------------------------
# Address blocking key
# ------------------------------------------------------------

def make_address_block(country, address):
    """
    Create a relatively compact address blocking key.

    We use:
        country + first 12 normalized address characters

    Empty/missing addresses are excluded from blocking.
    """

    if pd.isna(address):
        return None

    address = normalize_address(str(address))

    if not address:
        return None

    # Keep only the first part of the normalized address.
    prefix = address[:12]

    if not prefix:
        return None

    return f"{country}|{prefix}"


# ------------------------------------------------------------
# Load S1
# ------------------------------------------------------------

print("Loading S1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"S1 rows: {len(s1):,}")


# ------------------------------------------------------------
# Load ground truth
# ------------------------------------------------------------

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

# Build S1 -> true S2/S3 IDs
truth_s2 = defaultdict(set)
truth_s3 = defaultdict(set)

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]
    matched = row["matched_entity_ids"]

    if not matched:
        continue

    for entity_id in matched.split(","):

        if entity_id.startswith("S2-"):
            truth_s2[s1_id].add(entity_id)

        elif entity_id.startswith("S3-"):
            truth_s3[s1_id].add(entity_id)


# ------------------------------------------------------------
# Build S2 address index
# ------------------------------------------------------------

print("\nBuilding S2 address index...")

s2_index = defaultdict(list)

reader = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE
)

total = 0

for chunk in reader:

    for row in chunk.itertuples(index=False):

        block = make_address_block(
            row.country,
            row.business_address
        )

        if block is not None:
            s2_index[block].append(row.entity_id)

    total += len(chunk)

    if total % 1_000_000 == 0:
        print(f"S2 rows indexed: {total:,}")

print(f"S2 rows indexed: {total:,}")
print(f"S2 blocks: {len(s2_index):,}")


# ------------------------------------------------------------
# Build S3 address index
# ------------------------------------------------------------

print("\nBuilding S3 address index...")

s3_index = defaultdict(list)

reader = pd.read_csv(
    S3_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=CHUNK_SIZE
)

total = 0

for chunk in reader:

    for row in chunk.itertuples(index=False):

        block = make_address_block(
            row.country,
            row.business_address
        )

        if block is not None:
            s3_index[block].append(row.entity_id)

    total += len(chunk)

    if total % 1_000_000 == 0:
        print(f"S3 rows indexed: {total:,}")

print(f"S3 rows indexed: {total:,}")
print(f"S3 blocks: {len(s3_index):,}")


# ------------------------------------------------------------
# Evaluate S2
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATING S2")
print("=" * 60)

total_entities = 0
total_true_links = 0
preserved_links = 0
candidate_total = 0

for row in s1.itertuples(index=False):

    s1_id = row.entity_id

    true_ids = truth_s2.get(s1_id)

    if not true_ids:
        continue

    total_entities += 1
    total_true_links += len(true_ids)

    block = make_address_block(
        row.country,
        row.business_address
    )

    if block is None:
        continue

    candidates = s2_index.get(block, [])

    candidate_total += len(candidates)

    candidate_set = set(candidates)

    preserved_links += len(true_ids.intersection(candidate_set))


recall = (
    preserved_links / total_true_links
    if total_true_links
    else 0
)

avg_candidates = (
    candidate_total / total_entities
    if total_entities
    else 0
)

print(f"Entities with matches: {total_entities:,}")
print(f"True links: {total_true_links:,}")
print(f"True links preserved: {preserved_links:,}")
print(f"Blocking recall: {recall * 100:.4f}%")
print(f"Average candidates/entity: {avg_candidates:,.2f}")


# ------------------------------------------------------------
# Evaluate S3
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATING S3")
print("=" * 60)

total_entities = 0
total_true_links = 0
preserved_links = 0
candidate_total = 0

for row in s1.itertuples(index=False):

    s1_id = row.entity_id

    true_ids = truth_s3.get(s1_id)

    if not true_ids:
        continue

    total_entities += 1
    total_true_links += len(true_ids)

    block = make_address_block(
        row.country,
        row.business_address
    )

    if block is None:
        continue

    candidates = s3_index.get(block, [])

    candidate_total += len(candidates)

    candidate_set = set(candidates)

    preserved_links += len(true_ids.intersection(candidate_set))


recall = (
    preserved_links / total_true_links
    if total_true_links
    else 0
)

avg_candidates = (
    candidate_total / total_entities
    if total_entities
    else 0
)

print(f"Entities with matches: {total_entities:,}")
print(f"True links: {total_true_links:,}")
print(f"True links preserved: {preserved_links:,}")
print(f"Blocking recall: {recall * 100:.4f}%")
print(f"Average candidates/entity: {avg_candidates:,.2f}")

print("\nAddress blocking evaluation complete.")