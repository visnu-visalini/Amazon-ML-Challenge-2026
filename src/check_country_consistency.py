import pandas as pd
from pathlib import Path

BASE = Path("D:/Amazon ML Challenge/dataset/train")

GT_FILE = BASE / "train_ground_truth.tsv"
S1_FILE = BASE / "train_source1.tsv"
S2_FILE = BASE / "train_source2.tsv"
S3_FILE = BASE / "train_source3.tsv"


print("Loading S1 countries...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=["entity_id", "country"],
    dtype=str
)

s1["country"] = (
    s1["country"]
    .fillna("")
    .str.strip()
    .str.lower()
)

s1_country = dict(
    zip(s1["entity_id"], s1["country"])
)

print(f"S1 loaded: {len(s1):,}")


print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    usecols=["source1_entity_id", "matched_entity_ids"],
    dtype=str
)

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)

print(f"Ground truth loaded: {len(gt):,}")


# ---------------------------------------------------------
# Build reverse mapping:
#
# S2/S3 entity ID -> S1 entity ID
# ---------------------------------------------------------

print("\nBuilding match lookup...")

match_lookup = {}

for s1_id, matches in zip(
    gt["source1_entity_id"],
    gt["matched_entity_ids"]
):

    if not matches:
        continue

    for entity_id in matches.split(","):

        entity_id = entity_id.strip()

        if entity_id:
            match_lookup[entity_id] = s1_id


print(
    f"Total matched S2/S3 IDs: "
    f"{len(match_lookup):,}"
)


# ---------------------------------------------------------
# Check one source
# ---------------------------------------------------------

def check_source(filename, prefix):

    print("\n" + "=" * 60)
    print(f"CHECKING {prefix}")
    print("=" * 60)

    total = 0
    country_match = 0
    country_mismatch = 0

    needed_ids = {
        entity_id
        for entity_id in match_lookup
        if entity_id.startswith(prefix)
    }

    print(
        f"Ground-truth {prefix} IDs: "
        f"{len(needed_ids):,}"
    )

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        usecols=["entity_id", "country"],
        dtype=str,
        chunksize=200_000
    ):

        # Only keep IDs that occur in ground truth
        chunk = chunk[
            chunk["entity_id"].isin(needed_ids)
        ]

        for entity_id, country in zip(
            chunk["entity_id"],
            chunk["country"]
        ):

            s1_id = match_lookup.get(entity_id)

            if s1_id is None:
                continue

            s1_c = s1_country.get(s1_id, "")
            target_c = str(country).strip().lower()

            total += 1

            if s1_c == target_c:
                country_match += 1
            else:
                country_mismatch += 1

    print(f"Checked:          {total:,}")
    print(f"Country matches:  {country_match:,}")
    print(f"Country mismatch: {country_mismatch:,}")

    if total > 0:

        percentage = (
            country_match / total
        ) * 100

        print(
            f"Same-country rate: "
            f"{percentage:.4f}%"
        )

    return (
        total,
        country_match,
        country_mismatch
    )


s2_results = check_source(
    S2_FILE,
    "S2-"
)

s3_results = check_source(
    S3_FILE,
    "S3-"
)


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

total = s2_results[0] + s3_results[0]
matches = s2_results[1] + s3_results[1]
mismatches = s2_results[2] + s3_results[2]

print(f"Total links checked: {total:,}")
print(f"Same country:        {matches:,}")
print(f"Different country:   {mismatches:,}")

if total > 0:

    print(
        f"\nOverall same-country rate: "
        f"{matches / total * 100:.4f}%"
    )

    print(
        f"Overall mismatch rate: "
        f"{mismatches / total * 100:.4f}%"
    )