import re
import gc
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(r"D:\Amazon ML Challenge")
DATA = BASE / "dataset" / "test"
OUT = BASE / "output"

S1 = DATA / "test_source1.tsv"
S2 = DATA / "test_source2.tsv"
S3 = DATA / "test_source3.tsv"


def normalize(value):
    if not value:
        return ""

    value = str(value).lower().strip()

    # Keep only letters and numbers
    value = re.sub(r"[^a-z0-9]+", "", value)

    return value


def make_name_key(country, name):
    country = str(country).lower().strip()
    name = normalize(name)

    if not country or not name:
        return None

    return country + "|" + name


def make_address_key(country, address):
    country = str(country).lower().strip()
    address = normalize(address)

    if not country or not address:
        return None

    return country + "|" + address


print("=" * 70)
print("FAST EXACT-MATCH SUBMISSION")
print("=" * 70)

# ------------------------------------------------------------
# LOAD S1
# ------------------------------------------------------------

print("\nLoading S1...")

s1 = pd.read_csv(
    S1,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"S1 rows: {len(s1):,}")

s1_ids = s1["entity_id"].tolist()

# ------------------------------------------------------------
# BUILD EXACT INDEXES
# ------------------------------------------------------------

print("\nBuilding exact indexes...")

name_index = defaultdict(list)
address_index = defaultdict(list)

for i, row in enumerate(
    s1[["entity_id", "business_name", "business_address", "country"]].itertuples(index=False)
):

    entity_id, name, address, country = row

    nk = make_name_key(country, name)
    if nk:
        name_index[nk].append(entity_id)

    ak = make_address_key(country, address)
    if ak:
        address_index[ak].append(entity_id)

    if (i + 1) % 500000 == 0:
        print(f"Indexed S1: {i + 1:,}/{len(s1):,}", flush=True)

print(f"Name keys   : {len(name_index):,}")
print(f"Address keys: {len(address_index):,}")

# ------------------------------------------------------------
# MATCH STORAGE
# ------------------------------------------------------------

matches = defaultdict(set)

# ------------------------------------------------------------
# PROCESS SOURCE
# ------------------------------------------------------------

def process_source(path, label):

    print("\n" + "=" * 70)
    print(f"PROCESSING {label}")
    print("=" * 70)

    processed = 0
    name_hits = 0
    address_hits = 0
    total_matches = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100000
    ):

        for row in chunk.itertuples(index=False):

            src_id = row.entity_id
            name = row.business_name
            address = row.business_address
            country = row.country

            found = set()

            nk = make_name_key(country, name)

            if nk:
                ids = name_index.get(nk)
                if ids:
                    found.update(ids)
                    name_hits += 1

            ak = make_address_key(country, address)

            if ak:
                ids = address_index.get(ak)
                if ids:
                    before = len(found)
                    found.update(ids)

                    if len(ids) > 0:
                        address_hits += 1

            if found:
                for s1_id in found:
                    matches[s1_id].add(src_id)

                total_matches += len(found)

            processed += 1

        print(
            f"{label}: {processed:,} | "
            f"name hits: {name_hits:,} | "
            f"address hits: {address_hits:,} | "
            f"matches: {total_matches:,}",
            flush=True
        )

        del chunk
        gc.collect()

    print(f"\n{label} complete.")
    print(f"Rows processed: {processed:,}")
    print(f"Matches found : {total_matches:,}")


# ------------------------------------------------------------
# S2 + S3
# ------------------------------------------------------------

process_source(S2, "S2")
process_source(S3, "S3")

# ------------------------------------------------------------
# WRITE SUBMISSION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WRITING SUBMISSION")
print("=" * 70)

OUT.mkdir(parents=True, exist_ok=True)

output_file = OUT / "matching_results.tsv"

with open(output_file, "w", encoding="utf-8", newline="") as f:

    f.write("source1_entity_id\tmatched_entity_ids\n")

    matched_entities = 0

    for i, s1_id in enumerate(s1_ids):

        result = matches.get(s1_id, set())

        if result:
            matched_entities += 1
            value = ",".join(sorted(result))
        else:
            value = ""

        f.write(f"{s1_id}\t{value}\n")

        if (i + 1) % 500000 == 0:
            print(
                f"Writing: {i + 1:,}/{len(s1_ids):,}",
                flush=True
            )

print("\n" + "=" * 70)
print("SUBMISSION READY")
print("=" * 70)

print(f"File: {output_file}")
print(f"S1 rows: {len(s1_ids):,}")
print(f"S1 entities with matches: {matched_entities:,}")
print(f"File exists: {output_file.exists()}")