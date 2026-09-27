import pandas as pd
from pathlib import Path
import re
import unicodedata


BASE = Path("D:/Amazon ML Challenge/dataset/train")

GT_FILE = BASE / "train_ground_truth.tsv"
S1_FILE = BASE / "train_source1.tsv"
S2_FILE = BASE / "train_source2.tsv"
S3_FILE = BASE / "train_source3.tsv"


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def normalize_name(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def blocking_key(name, country):

    name = normalize_name(name)

    country = (
        str(country)
        .strip()
        .lower()
    )

    if not name:
        name_prefix = "#"
    else:
        name_prefix = name[0]

    return country + "|" + name_prefix


# ---------------------------------------------------------
# Load S1
# ---------------------------------------------------------

print("Loading S1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "country"
    ],
    dtype=str
)

print(
    f"S1 rows: {len(s1):,}"
)


# ---------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    usecols=[
        "source1_entity_id",
        "matched_entity_ids"
    ],
    dtype=str
)

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)


# ---------------------------------------------------------
# Build TRUE match lookup
# ---------------------------------------------------------

print("\nBuilding ground-truth lookup...")

true_matches = {}

for s1_id, matches in zip(
    gt["source1_entity_id"],
    gt["matched_entity_ids"]
):

    if not matches:
        true_matches[s1_id] = set()
        continue

    true_matches[s1_id] = {
        x.strip()
        for x in matches.split(",")
        if x.strip()
    }


# ---------------------------------------------------------
# Build S2/S3 blocking index
# ---------------------------------------------------------

def build_index(filename, source_name):

    print(
        f"\nBuilding {source_name} index..."
    )

    index = {}

    total = 0

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "country"
        ],
        dtype=str,
        chunksize=200_000
    ):

        for entity_id, name, country in zip(
            chunk["entity_id"],
            chunk["business_name"],
            chunk["country"]
        ):

            key = blocking_key(
                name,
                country
            )

            if key not in index:
                index[key] = set()

            index[key].add(entity_id)

            total += 1

    print(
        f"{source_name} rows indexed: "
        f"{total:,}"
    )

    print(
        f"{source_name} blocks: "
        f"{len(index):,}"
    )

    return index


s2_index = build_index(
    S2_FILE,
    "S2"
)

s3_index = build_index(
    S3_FILE,
    "S3"
)


# ---------------------------------------------------------
# Evaluate blocking
# ---------------------------------------------------------

def evaluate(
    source_index,
    source_name
):

    print("\n" + "=" * 60)
    print(
        f"EVALUATING {source_name}"
    )
    print("=" * 60)

    total_true = 0
    preserved_true = 0

    entities_with_matches = 0

    candidate_total = 0

    for _, row in s1.iterrows():

        s1_id = row["entity_id"]

        matches = true_matches.get(
            s1_id,
            set()
        )

        source_matches = {
            x
            for x in matches
            if x.startswith(source_name + "-")
        }

        if not source_matches:
            continue

        entities_with_matches += 1

        total_true += len(source_matches)

        key = blocking_key(
            row["business_name"],
            row["country"]
        )

        candidates = source_index.get(
            key,
            set()
        )

        candidate_total += len(candidates)

        preserved = (
            source_matches
            & candidates
        )

        preserved_true += len(
            preserved
        )

    recall = (
        preserved_true / total_true
        if total_true
        else 0
    )

    avg_candidates = (
        candidate_total
        / entities_with_matches
        if entities_with_matches
        else 0
    )

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
        f"{avg_candidates:.2f}"
    )


evaluate(
    s2_index,
    "S2"
)

evaluate(
    s3_index,
    "S3"
)