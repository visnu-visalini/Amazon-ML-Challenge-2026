import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"


def load_ground_truth():
    return pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        keep_default_na=False
    )


def get_sample_ids(gt, sample_size=10):
    samples = []

    for _, row in gt.iterrows():

        matched = row["matched_entity_ids"]

        if not matched:
            continue

        ids = matched.split(",")

        samples.append(
            (
                row["source1_entity_id"],
                ids[:4]
            )
        )

        if len(samples) >= sample_size:
            break

    return samples


def find_records(file_path, ids, source_name):
    """
    Search only for required entity IDs using chunks.
    """

    results = []

    ids = set(ids)

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=200_000,
        dtype=str
    ):

        found = chunk[
            chunk["entity_id"].isin(ids)
        ]

        if not found.empty:
            results.append(found)

        if sum(len(x) for x in results) >= len(ids):
            break

    if results:
        return pd.concat(results, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )


def main():

    print("\n" + "=" * 80)
    print("MATCH PATTERN ANALYSIS")
    print("=" * 80)

    print("\nLoading ground truth...")

    gt = load_ground_truth()

    samples = get_sample_ids(gt, sample_size=10)

    s1_ids = [
        source1_id
        for source1_id, _ in samples
    ]

    matched_ids = []

    for _, ids in samples:
        matched_ids.extend(ids)

    s2_ids = [
        x for x in matched_ids
        if x.startswith("S2-")
    ]

    s3_ids = [
        x for x in matched_ids
        if x.startswith("S3-")
    ]

    print(f"\nSample S1 entities : {len(s1_ids)}")
    print(f"Sample S2 matches  : {len(s2_ids)}")
    print(f"Sample S3 matches  : {len(s3_ids)}")

    # ---------------------------------------------------------
    # FIND REQUIRED RECORDS
    # ---------------------------------------------------------

    print("\nSearching Source 1...")

    s1 = find_records(
        TRAIN_DIR / "train_source1.tsv",
        s1_ids,
        "S1"
    )

    print("Searching Source 2...")

    s2 = find_records(
        TRAIN_DIR / "train_source2.tsv",
        s2_ids,
        "S2"
    )

    print("Searching Source 3...")

    s3 = find_records(
        TRAIN_DIR / "train_source3.tsv",
        s3_ids,
        "S3"
    )

    # ---------------------------------------------------------
    # CREATE LOOKUPS
    # ---------------------------------------------------------

    s1_lookup = s1.set_index("entity_id")
    s2_lookup = s2.set_index("entity_id")
    s3_lookup = s3.set_index("entity_id")

    # ---------------------------------------------------------
    # PRINT MATCH EXAMPLES
    # ---------------------------------------------------------

    print("\n\nTRUE MATCH EXAMPLES")
    print("-" * 80)

    for source1_id, ids in samples:

        print("\n" + "=" * 80)

        row = s1_lookup.loc[source1_id]

        print("\nSOURCE 1")
        print("ID      :", source1_id)
        print("NAME    :", row["business_name"])
        print("ADDRESS :", row["business_address"])
        print("COUNTRY :", row["country"])

        print("\nTRUE MATCHES")

        for entity_id in ids:

            if entity_id.startswith("S2-"):
                lookup = s2_lookup
            else:
                lookup = s3_lookup

            if entity_id not in lookup.index:
                continue

            match = lookup.loc[entity_id]

            print("\nID      :", entity_id)
            print("NAME    :", match["business_name"])
            print("ADDRESS :", match["business_address"])
            print("COUNTRY :", match["country"])

    print("\n\nDONE.")


if __name__ == "__main__":
    main()