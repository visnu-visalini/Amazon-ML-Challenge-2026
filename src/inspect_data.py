import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"


def inspect_file(path):
    print("\n" + "=" * 70)
    print(f"FILE: {path.name}")
    print("=" * 70)

    df = pd.read_csv(path, sep="\t")

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

    return df


def main():

    train_files = [
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
        TRAIN_DIR / "train_ground_truth.tsv",
    ]

    test_files = [
        TEST_DIR / "test_source1.tsv",
        TEST_DIR / "test_source2.tsv",
        TEST_DIR / "test_source3.tsv",
    ]

    print("\n" + "#" * 70)
    print("AMAZON ML CHALLENGE 2026 - DATASET INSPECTION")
    print("#" * 70)

    print("\n\nTRAINING DATA")
    print("-" * 70)

    for file in train_files:
        inspect_file(file)

    print("\n\nTEST DATA")
    print("-" * 70)

    for file in test_files:
        inspect_file(file)


if __name__ == "__main__":
    main()