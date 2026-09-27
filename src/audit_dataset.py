import pandas as pd
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"


def count_rows_and_countries(file_path, chunksize=200_000):
    """
    Read a TSV in chunks so we don't load huge files into memory.
    """

    total_rows = 0
    country_counts = Counter()

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        chunksize=chunksize
    ):
        total_rows += len(chunk)
        country_counts.update(chunk["country"].dropna())

    return total_rows, country_counts


def audit_ground_truth(file_path, chunksize=200_000):
    """
    Analyze how many matches each Source 1 entity has.
    """

    match_distribution = Counter()

    total_rows = 0
    empty_matches = 0
    non_empty_matches = 0

    s2_matches = 0
    s3_matches = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
        chunksize=chunksize,
        keep_default_na=False
    ):

        total_rows += len(chunk)

        for value in chunk["matched_entity_ids"]:

            # Empty match list
            if value == "":
                empty_matches += 1
                match_distribution[0] += 1
                continue

            ids = value.split(",")

            count = len(ids)

            non_empty_matches += 1
            match_distribution[count] += 1

            for entity_id in ids:
                if entity_id.startswith("S2-"):
                    s2_matches += 1
                elif entity_id.startswith("S3-"):
                    s3_matches += 1

    return {
        "total_rows": total_rows,
        "empty_matches": empty_matches,
        "non_empty_matches": non_empty_matches,
        "match_distribution": match_distribution,
        "s2_matches": s2_matches,
        "s3_matches": s3_matches,
    }


def main():

    print("\n" + "=" * 80)
    print("AMAZON ML CHALLENGE 2026 - DATASET AUDIT")
    print("=" * 80)

    # ---------------------------------------------------------
    # COUNTRY DISTRIBUTION
    # ---------------------------------------------------------

    print("\n\nCOUNTRY DISTRIBUTION")
    print("-" * 80)

    files = [
        ("TRAIN S1", TRAIN_DIR / "train_source1.tsv"),
        ("TRAIN S2", TRAIN_DIR / "train_source2.tsv"),
        ("TRAIN S3", TRAIN_DIR / "train_source3.tsv"),
        ("TEST S1", TEST_DIR / "test_source1.tsv"),
        ("TEST S2", TEST_DIR / "test_source2.tsv"),
        ("TEST S3", TEST_DIR / "test_source3.tsv"),
    ]

    for name, file_path in files:

        total, countries = count_rows_and_countries(file_path)

        print(f"\n{name}")
        print(f"Total rows: {total:,}")

        for country, count in countries.most_common():
            percentage = count / total * 100
            print(f"  {country:10s}: {count:10,} ({percentage:6.2f}%)")

    # ---------------------------------------------------------
    # GROUND TRUTH
    # ---------------------------------------------------------

    print("\n\nGROUND TRUTH ANALYSIS")
    print("-" * 80)

    result = audit_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    print(f"\nTotal Source 1 entities : {result['total_rows']:,}")

    print(f"Singletons / no matches : {result['empty_matches']:,}")

    print(f"Entities with matches   : {result['non_empty_matches']:,}")

    singleton_percentage = (
        result["empty_matches"]
        / result["total_rows"]
        * 100
    )

    print(
        f"Singleton percentage    : "
        f"{singleton_percentage:.2f}%"
    )

    print("\nMatch count distribution:")
    print("-" * 50)

    for count, frequency in sorted(
        result["match_distribution"].items()
    ):
        percentage = frequency / result["total_rows"] * 100

        print(
            f"{count:3d} matches : "
            f"{frequency:10,} entities "
            f"({percentage:6.2f}%)"
        )

    print("\nTotal ground-truth links:")
    print(f"S2 links: {result['s2_matches']:,}")
    print(f"S3 links: {result['s3_matches']:,}")
    print(
        f"Total   : "
        f"{result['s2_matches'] + result['s3_matches']:,}"
    )


if __name__ == "__main__":
    main()