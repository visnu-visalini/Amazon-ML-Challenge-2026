import gc
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from rapidfuzz import fuzz, process

from similarity_features import calculate_features


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"D:\Amazon ML Challenge")

DATASET_DIR = BASE_DIR / "dataset"
MODEL_FILE = BASE_DIR / "models" / "entity_matcher.txt"

S1_FILE = DATASET_DIR / "test" / "test_source1.tsv"
S2_FILE = DATASET_DIR / "test" / "test_source2.tsv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "s2_matches_test.csv"

THRESHOLD = 0.80

# SAFETY RUN ONLY
MAX_SOURCE2_ROWS = 10_000

# Cheap first-stage name filter.
# Address12 candidates are NOT filtered.
NAME_PREFILTER_SCORE = 35.0

# Number of feature rows sent to LightGBM at once.
FEATURE_BATCH_SIZE = 10_000


FEATURES = [
    "name_token_sort",
    "name_token_set",
    "name_character",
    "name_partial",
    "name_jaccard",
    "address_token_sort",
    "address_token_set",
    "address_character",
    "address_partial",
    "address_jaccard",
    "address_digit_similarity",
    "country_match",
]


# ============================================================
# BLOCK NORMALIZATION
# ============================================================

def block_normalize(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    chars = []
    previous_space = False

    for ch in value:
        if ch.isalnum():
            chars.append(ch)
            previous_space = False
        else:
            if not previous_space:
                chars.append(" ")
                previous_space = True

    return "".join(chars).strip()


def compact_name(value):
    return block_normalize(value).replace(" ", "")


def make_name_key(country, name):
    name_norm = block_normalize(name)

    if not name_norm:
        return None

    return f"{str(country).lower().strip()}|{name_norm[:2]}"


def make_address_key(country, address):
    address_norm = block_normalize(address)

    if not address_norm:
        return None

    return f"{str(country).lower().strip()}|{address_norm[:12]}"


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = lgb.Booster(model_file=str(MODEL_FILE))

print("Model loaded.")
print("Threshold:", THRESHOLD)
print("Name prefilter:", NAME_PREFILTER_SCORE)


# ============================================================
# LOAD TEST SOURCE 1
# ============================================================

print()
print("=" * 70)
print("LOADING TEST SOURCE 1")
print("=" * 70)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("S1 rows:", len(s1))


# Convert columns to lists.
# Indexes will store integer positions instead of duplicating
# complete source-1 records.

s1_ids = s1["entity_id"].tolist()
s1_names = s1["business_name"].tolist()
s1_addresses = s1["business_address"].tolist()
s1_countries = s1["country"].tolist()


# ============================================================
# BUILD BLOCK INDEXES
# ============================================================

print()
print("=" * 70)
print("BUILDING BLOCK INDEXES")
print("=" * 70)

name_index = {}
address_index = {}

for pos in range(len(s1_ids)):

    country = s1_countries[pos]
    name = s1_names[pos]
    address = s1_addresses[pos]

    # -------------------------
    # NAME2 INDEX
    # -------------------------

    name_key = make_name_key(country, name)

    if name_key is not None:

        if name_key not in name_index:
            name_index[name_key] = ([], [])

        name_names, name_positions = name_index[name_key]

        name_names.append(compact_name(name))
        name_positions.append(pos)

    # -------------------------
    # ADDRESS12 INDEX
    # -------------------------

    address_key = make_address_key(country, address)

    if address_key is not None:

        if address_key not in address_index:
            address_index[address_key] = []

        address_index[address_key].append(pos)


print("Name blocks   :", len(name_index))
print("Address blocks:", len(address_index))


# ============================================================
# LOAD TEST SOURCE 2
# ============================================================

print()
print("=" * 70)
print("LOADING TEST SOURCE 2")
print("=" * 70)

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=MAX_SOURCE2_ROWS
)

print("S2 rows for safety run:", len(s2))


# ============================================================
# OUTPUT SETUP
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

output_started = False


def write_matches(rows):
    global output_started

    if not rows:
        return

    df = pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "source2_entity_id",
            "score",
        ]
    )

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not output_started,
        index=False
    )

    output_started = True


# ============================================================
# PREDICTION BATCH
# ============================================================

feature_buffer = []
pair_buffer = []

total_model_candidates = 0
total_matches = 0
prediction_batch = 0


def run_prediction_batch():
    global total_matches
    global prediction_batch

    if not feature_buffer:
        return

    prediction_batch += 1

    X = pd.DataFrame(feature_buffer)

    X = X[FEATURES]

    probabilities = model.predict(X)

    accepted = []

    for i, probability in enumerate(probabilities):

        if probability >= THRESHOLD:

            s2_id, s1_pos = pair_buffer[i]

            accepted.append(
                (
                    s1_ids[s1_pos],
                    s2_id,
                    float(probability)
                )
            )

    if accepted:
        write_matches(accepted)

        total_matches += len(accepted)

    print(
        f"Prediction batch {prediction_batch}: "
        f"{len(feature_buffer):,} candidates | "
        f"{len(accepted):,} matches"
    )

    feature_buffer.clear()
    pair_buffer.clear()

    del X
    del probabilities
    del accepted

    gc.collect()


# ============================================================
# MAIN PREDICTION LOOP
# ============================================================

print()
print("=" * 70)
print("STARTING OPTIMIZED TEST PREDICTION")
print("=" * 70)

print()
print("IMPORTANT:")
print("- Address12 candidates go directly to the model.")
print("- Name2 candidates use a cheap RapidFuzz pre-filter.")
print("- No top-1 selection is used.")
print("- All qualifying matches >= 0.80 are retained.")
print()


for row_number, row in enumerate(s2.itertuples(index=False), start=1):

    s2_id = row.entity_id
    s2_name = row.business_name
    s2_address = row.business_address
    s2_country = row.country

    # --------------------------------------------------------
    # ADDRESS12 CANDIDATES
    # --------------------------------------------------------

    address_key = make_address_key(
        s2_country,
        s2_address
    )

    address_positions = address_index.get(
        address_key,
        []
    )

    # Address candidates bypass the cheap name filter.
    candidate_positions = set(address_positions)

    # --------------------------------------------------------
    # NAME2 CANDIDATES
    # --------------------------------------------------------

    name_key = make_name_key(
        s2_country,
        s2_name
    )

    name_prefilter_survivors = 0

    if name_key is not None:

        name_bucket = name_index.get(name_key)

        if name_bucket is not None:

            candidate_names, candidate_positions_list = name_bucket

            query_name = compact_name(s2_name)

            if query_name:

                filtered = process.extract(
                    query_name,
                    candidate_names,
                    scorer=fuzz.ratio,
                    score_cutoff=NAME_PREFILTER_SCORE,
                    limit=None
                )

                for _, score, local_index in filtered:

                    s1_position = candidate_positions_list[local_index]

                    candidate_positions.add(s1_position)

                    name_prefilter_survivors += 1

    # --------------------------------------------------------
    # NO CANDIDATES
    # --------------------------------------------------------

    if not candidate_positions:
        continue

    # --------------------------------------------------------
    # FULL MODEL FEATURES
    # --------------------------------------------------------

    for s1_position in candidate_positions:

        features = calculate_features(
            s1_names[s1_position],
            s1_addresses[s1_position],
            s1_countries[s1_position],
            s2_name,
            s2_address,
            s2_country
        )

        feature_buffer.append(features)

        pair_buffer.append(
            (
                s2_id,
                s1_position
            )
        )

        total_model_candidates += 1

        if len(feature_buffer) >= FEATURE_BATCH_SIZE:
            run_prediction_batch()

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if row_number % 500 == 0:

        print(
            f"Processed S2 rows: {row_number:,} / {len(s2):,} | "
            f"Model candidates: {total_model_candidates:,} | "
            f"Matches: {total_matches:,}"
        )


# ============================================================
# FINAL BATCH
# ============================================================

run_prediction_batch()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("OPTIMIZED TEST PREDICTION COMPLETE")
print("=" * 70)

print("S2 rows processed     :", len(s2))
print("Model candidates      :", f"{total_model_candidates:,}")
print("Matches >= 0.80       :", f"{total_matches:,}")
print("Output file            :", OUTPUT_FILE)

if OUTPUT_FILE.exists():
    print("Output size (MB)       :",
          round(OUTPUT_FILE.stat().st_size / (1024 * 1024), 2))

print()
print("This was a 10K safety run.")
print("Do NOT scale to the full test set yet.")