from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz

from normalization import (
    normalize_business_name,
    normalize_address,
    normalize_alphanumeric,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"D:\Amazon ML Challenge")

MODEL_PATH = BASE_DIR / "models" / "entity_matcher.txt"
GATE_PATH = BASE_DIR / "models" / "fast_candidate_gate.txt"

S1_PATH = BASE_DIR / "dataset" / "source1.csv"
S2_PATH = BASE_DIR / "dataset" / "source2.csv"

MAX_SOURCE2_ROWS = 10_000

GATE_THRESHOLD = 0.10
MODEL_THRESHOLD = 0.80

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

FAST_FEATURES = [
    "name_token_sort",
    "name_token_set",
    "name_character",
    "name_jaccard",
    "address_token_sort",
    "address_token_set",
    "address_character",
    "address_jaccard",
    "address_digit_similarity",
    "country_match",
]


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("FAST GATE SAFETY TEST")
print("=" * 70)

print("\nLoading models...")

model = lgb.Booster(
    model_file=str(MODEL_PATH)
)

gate = lgb.Booster(
    model_file=str(GATE_PATH)
)

print("Main model loaded.")
print("Fast gate loaded.")


# ============================================================
# LOAD S1
# ============================================================

print("\nLoading Source 1...")

s1 = pd.read_csv(S1_PATH)

s1_names = (
    s1["business_name"]
    .fillna("")
    .astype(str)
    .map(normalize_business_name)
    .tolist()
)

s1_addresses = (
    s1["business_address"]
    .fillna("")
    .astype(str)
    .map(normalize_address)
    .tolist()
)

s1_countries = (
    s1["country"]
    .fillna("")
    .astype(str)
    .tolist()
)

print("S1 entities:", f"{len(s1):,}")


# ============================================================
# LOAD S2 SAMPLE
# ============================================================

print("\nLoading Source 2 sample...")

s2 = pd.read_csv(
    S2_PATH,
    nrows=MAX_SOURCE2_ROWS
)

print("S2 rows:", f"{len(s2):,}")


# ============================================================
# BUILD BLOCK INDEXES
# ============================================================

print("\nBuilding blocking indexes...")

name_index = {}
address_index = {}

for i, (name, address, country) in enumerate(
    zip(
        s1_names,
        s1_addresses,
        s1_countries
    )
):

    country_key = country.strip().lower()

    # name2 block
    name_key = (
        country_key,
        normalize_alphanumeric(name)[:2]
    )

    name_index.setdefault(
        name_key,
        []
    ).append(i)

    # address12 block
    address_key = (
        country_key,
        normalize_alphanumeric(address)[:12]
    )

    address_index.setdefault(
        address_key,
        []
    ).append(i)


print("Name blocks:", f"{len(name_index):,}")
print("Address blocks:", f"{len(address_index):,}")


# ============================================================
# PROCESS
# ============================================================

total_candidates = 0
total_gate_candidates = 0
total_final_candidates = 0

rows_without_candidates = 0

print("\nStarting benchmark...")
print("Gate threshold :", GATE_THRESHOLD)
print("Model threshold:", MODEL_THRESHOLD)

for row_number, row in s2.iterrows():

    s2_name = normalize_business_name(
        str(row["business_name"])
        if pd.notna(row["business_name"])
        else ""
    )

    s2_address = normalize_address(
        str(row["business_address"])
        if pd.notna(row["business_address"])
        else ""
    )

    s2_country = (
        str(row["country"])
        if pd.notna(row["country"])
        else ""
    )

    country_key = s2_country.strip().lower()

    candidate_set = set()

    # --------------------------------------------------------
    # NAME2 BLOCK
    # --------------------------------------------------------

    name_key = (
        country_key,
        normalize_alphanumeric(s2_name)[:2]
    )

    candidate_set.update(
        name_index.get(name_key, [])
    )

    # --------------------------------------------------------
    # ADDRESS12 BLOCK
    # --------------------------------------------------------

    address_key = (
        country_key,
        normalize_alphanumeric(s2_address)[:12]
    )

    candidate_set.update(
        address_index.get(address_key, [])
    )

    candidates = list(candidate_set)

    total_candidates += len(candidates)

    if not candidates:
        rows_without_candidates += 1
        continue

    # --------------------------------------------------------
    # FAST GATE FEATURES
    # --------------------------------------------------------

    fast_rows = []

    for idx in candidates:

        s1_name = s1_names[idx]
        s1_address = s1_addresses[idx]
        s1_country = s1_countries[idx]

        name_token_sort = fuzz.token_sort_ratio(
            s1_name,
            s2_name
        )

        name_token_set = fuzz.token_set_ratio(
            s1_name,
            s2_name
        )

        name_character = fuzz.ratio(
            s1_name,
            s2_name
        )

        name_jaccard = (
            len(
                set(s1_name.split())
                & set(s2_name.split())
            )
            /
            max(
                len(
                    set(s1_name.split())
                    | set(s2_name.split())
                ),
                1
            )
        )

        address_token_sort = fuzz.token_sort_ratio(
            s1_address,
            s2_address
        )

        address_token_set = fuzz.token_set_ratio(
            s1_address,
            s2_address
        )

        address_character = fuzz.ratio(
            s1_address,
            s2_address
        )

        address_jaccard = (
            len(
                set(s1_address.split())
                & set(s2_address.split())
            )
            /
            max(
                len(
                    set(s1_address.split())
                    | set(s2_address.split())
                ),
                1
            )
        )

        s1_digits = "".join(
            c for c in s1_address if c.isdigit()
        )

        s2_digits = "".join(
            c for c in s2_address if c.isdigit()
        )

        if not s1_digits and not s2_digits:
            digit_similarity = 1.0
        elif not s1_digits or not s2_digits:
            digit_similarity = 0.0
        else:
            digit_similarity = (
                fuzz.ratio(
                    s1_digits,
                    s2_digits
                ) / 100.0
            )

        country_match = (
            1.0
            if s1_country.strip().lower()
            == s2_country.strip().lower()
            else 0.0
        )

        fast_rows.append([
            name_token_sort,
            name_token_set,
            name_character,
            name_jaccard,
            address_token_sort,
            address_token_set,
            address_character,
            address_jaccard,
            digit_similarity,
            country_match,
        ])

    fast_df = pd.DataFrame(
        fast_rows,
        columns=FAST_FEATURES
    )

    # --------------------------------------------------------
    # FAST GATE
    # --------------------------------------------------------

    gate_probability = gate.predict(
        fast_df
    )

    gate_mask = (
        gate_probability >= GATE_THRESHOLD
    )

    gated_positions = np.flatnonzero(
        gate_mask
    )

    total_gate_candidates += len(
        gated_positions
    )

    # --------------------------------------------------------
    # FULL MODEL
    # --------------------------------------------------------

    # For this safety test we only count how many candidates
    # reach the expensive full model.
    total_final_candidates += len(
        gated_positions
    )

    if row_number % 1000 == 0:
        print(
            f"Processed {row_number:,} / "
            f"{len(s2):,} | "
            f"raw={total_candidates:,} | "
            f"after_gate={total_gate_candidates:,}"
        )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("FAST GATE BENCHMARK RESULT")
print("=" * 70)

print(
    "S2 rows processed:",
    f"{len(s2):,}"
)

print(
    "Rows without candidates:",
    f"{rows_without_candidates:,}"
)

print(
    "Raw candidates:",
    f"{total_candidates:,}"
)

print(
    "Candidates after gate:",
    f"{total_gate_candidates:,}"
)

if total_candidates:
    reduction = (
        1
        -
        total_gate_candidates / total_candidates
    ) * 100

    average_raw = (
        total_candidates / len(s2)
    )

    average_gate = (
        total_gate_candidates / len(s2)
    )

    print(
        "Candidate reduction:",
        f"{reduction:.2f}%"
    )

    print(
        "Average raw candidates/S2:",
        f"{average_raw:.2f}"
    )

    print(
        "Average gated candidates/S2:",
        f"{average_gate:.2f}"
    )

print()
print("This was ONLY a benchmark.")
print("No output submission file was created.")