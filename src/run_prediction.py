"""
Production prediction for Amazon ML Challenge 2026.
Outputs matching_results.tsv and candidate_pairs.tsv.

Pipeline:
  name2 OR address12 blocking
  -> fast gate >= 0.10
  -> full 12-feature calculation
  -> main model >= 0.80
  -> retain ALL qualifying matches (no top-1)
"""

import gc
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import pandas as pd

# Add src to path so imports work when run from any directory
sys.path.insert(0, str(Path(__file__).parent))
from similarity_features import calculate_features

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"D:\Amazon ML Challenge")
DATASET_DIR = BASE_DIR / "dataset" / "test"
MODEL_FILE = BASE_DIR / "models" / "entity_matcher.txt"
GATE_FILE = BASE_DIR / "models" / "fast_candidate_gate.txt"
OUTPUT_DIR = BASE_DIR / "output"

S1_FILE = DATASET_DIR / "test_source1.tsv"
S2_FILE = DATASET_DIR / "test_source2.tsv"
S3_FILE = DATASET_DIR / "test_source3.tsv"

MAIN_THRESHOLD = 0.80
GATE_THRESHOLD = 0.10

# Batch size for model inference
FEATURE_BATCH_SIZE = 20_000

# Set to None for full run, or an integer for a test run
# e.g. TEST_LIMIT = 50_000 to test with first 50K S2/S3 rows
TEST_LIMIT = None  # Full run, process all rows

GATE_FEATURES = [
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

MAIN_FEATURES = [
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
# BLOCKING HELPERS
# ============================================================

def _block_norm(value):
    if not value or value != value:  # handles None and NaN
        return ""
    s = str(value).lower().strip()
    chars = []
    prev_space = False
    for ch in s:
        if ch.isalnum():
            chars.append(ch)
            prev_space = False
        elif not prev_space:
            chars.append(" ")
            prev_space = True
    return "".join(chars).strip()


def name_key(country, name):
    n = _block_norm(name).replace(" ", "")
    if not n:
        return None
    return f"{str(country).lower().strip()}|{n[:2]}"


def address_key(country, address):
    a = _block_norm(address)
    if not a:
        return None
    return f"{str(country).lower().strip()}|{a[:12]}"


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")
main_model = lgb.Booster(model_file=str(MODEL_FILE))
gate_model = lgb.Booster(model_file=str(GATE_FILE))
print(f"  Main model threshold : {MAIN_THRESHOLD}")
print(f"  Gate model threshold : {GATE_THRESHOLD}")

# ============================================================
# LOAD S1
# ============================================================

print("\nLoading S1...")
s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
print(f"  S1 rows: {len(s1):,}")

s1_ids = s1["entity_id"].tolist()
s1_names = s1["business_name"].tolist()
s1_addresses = s1["business_address"].tolist()
s1_countries = s1["country"].tolist()

# ============================================================
# BUILD S1 BLOCK INDEXES
# ============================================================

print("Building S1 block indexes...")
name_idx = {}    # key -> list of s1 positions
address_idx = {} # key -> list of s1 positions

for pos in range(len(s1_ids)):
    nk = name_key(s1_countries[pos], s1_names[pos])
    if nk is not None:
        name_idx.setdefault(nk, []).append(pos)

    ak = address_key(s1_countries[pos], s1_addresses[pos])
    if ak is not None:
        address_idx.setdefault(ak, []).append(pos)

print(f"  Name blocks   : {len(name_idx):,}")
print(f"  Address blocks: {len(address_idx):,}")

# ============================================================
# RESULT ACCUMULATORS
# ============================================================

# s1_matches[s1_id] = set of matched S2/S3 ids
s1_matches = defaultdict(set)
# s1_candidates[s1_id] = set of candidate S2/S3 ids (before main model)
s1_candidates = defaultdict(set)

# ============================================================
# PREDICTION ENGINE
# ============================================================

def process_source(source_file, source_label):
    """Process one source file (S2 or S3) and accumulate results."""

    print(f"\n{'='*60}")
    print(f"PROCESSING {source_label}")
    print(f"{'='*60}")

    df = pd.read_csv(
        source_file, sep="\t", dtype=str, keep_default_na=False,
        nrows=TEST_LIMIT
    )
    print(f"  Rows loaded: {len(df):,}")

    feature_buf = []
    pair_buf = []   # (src_id, s1_pos)

    total_candidates = 0
    total_gate_pass = 0
    total_matches = 0
    processed = 0

    def flush_batch():
        nonlocal total_gate_pass, total_matches

        if not feature_buf:
            return

        X_all = pd.DataFrame(feature_buf)

        # --- Fast gate (10 features) ---
        gate_probs = gate_model.predict(X_all[GATE_FEATURES])
        gate_mask = gate_probs >= GATE_THRESHOLD

        surviving_indices = [i for i, ok in enumerate(gate_mask) if ok]
        total_gate_pass += len(surviving_indices)

        if surviving_indices:
            # --- Main model (12 features) ---
            X_main = X_all.iloc[surviving_indices][MAIN_FEATURES]
            main_probs = main_model.predict(X_main)
            for j, prob in enumerate(main_probs):
                src_id, s1_pos = pair_buf[surviving_indices[j]]
                s1_id = s1_ids[s1_pos]
                s1_candidates[s1_id].add(src_id)
                if prob >= MAIN_THRESHOLD:
                    s1_matches[s1_id].add(src_id)
                    total_matches += 1

        feature_buf.clear()
        pair_buf.clear()

        del X_all, gate_probs, gate_mask
        gc.collect()

    for row in df.itertuples(index=False):
        src_id = row.entity_id
        src_name = row.business_name
        src_address = row.business_address
        src_country = row.country

        # Collect candidate S1 positions (union of name2 + address12)
        candidate_positions = set()

        nk = name_key(src_country, src_name)
        if nk is not None:
            for pos in name_idx.get(nk, []):
                candidate_positions.add(pos)

        ak = address_key(src_country, src_address)
        if ak is not None:
            for pos in address_idx.get(ak, []):
                candidate_positions.add(pos)

        total_candidates += len(candidate_positions)

        for s1_pos in candidate_positions:
            feats = calculate_features(
                s1_names[s1_pos], s1_addresses[s1_pos], s1_countries[s1_pos],
                src_name, src_address, src_country
            )
            feature_buf.append(feats)
            pair_buf.append((src_id, s1_pos))

            if len(feature_buf) >= FEATURE_BATCH_SIZE:
                flush_batch()

        processed += 1
        if processed % 100_000 == 0:
            print(
                f"  {source_label} processed: {processed:,}/{len(df):,} | "
                f"candidates: {total_candidates:,} | "
                f"gate pass: {total_gate_pass:,} | "
                f"matches: {total_matches:,}"
            )

    flush_batch()

    print(f"\n  {source_label} DONE")
    print(f"  Rows processed  : {processed:,}")
    print(f"  Total candidates: {total_candidates:,}")
    print(f"  Gate survivors  : {total_gate_pass:,}")
    print(f"  Matches >= {MAIN_THRESHOLD}: {total_matches:,}")


# ============================================================
# RUN S2 AND S3
# ============================================================

process_source(S2_FILE, "S2")
# Write partial results after S2 processing (checkpoint)

def _write_partial(prefix):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_rows = []
    candidate_rows = []
    for s1_id in s1_ids:
        matched = s1_matches.get(s1_id, set())
        candidates = s1_candidates.get(s1_id, set()) | matched
        matching_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(sorted(matched))})
        candidate_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(sorted(candidates))})
    pd.DataFrame(matching_rows).to_csv(OUTPUT_DIR / f"{prefix}_matching_results.tsv", sep="\t", index=False)
    pd.DataFrame(candidate_rows).to_csv(OUTPUT_DIR / f"{prefix}_candidate_pairs.tsv", sep="\t", index=False)
    print(f"Partial files written: {prefix}_matching_results.tsv, {prefix}_candidate_pairs.tsv")

_write_partial("s2_partial")
process_source(S3_FILE, "S3")

# ============================================================
# BUILD SUBMISSION FILES
# ============================================================

print(f"\n{'='*60}")
print("BUILDING SUBMISSION FILES")
print(f"{'='*60}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Every S1 entity must appear exactly once
matching_rows = []
candidate_rows = []

for s1_id in s1_ids:
    matched = s1_matches.get(s1_id, set())
    candidates = s1_candidates.get(s1_id, set())

    # Ensure all matches are also in candidates
    candidates = candidates | matched

    matched_str = ",".join(sorted(matched))
    candidate_str = ",".join(sorted(candidates))

    matching_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})
    candidate_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": candidate_str})

matching_df = pd.DataFrame(matching_rows)
candidate_df = pd.DataFrame(candidate_rows)

matching_file = OUTPUT_DIR / "matching_results.tsv"
candidate_file = OUTPUT_DIR / "candidate_pairs.tsv"

matching_df.to_csv(matching_file, sep="\t", index=False)
candidate_df.to_csv(candidate_file, sep="\t", index=False)

print(f"  matching_results.tsv : {len(matching_df):,} rows")
print(f"  candidate_pairs.tsv  : {len(candidate_df):,} rows")
print(f"  S1 entities with matches   : {(matching_df['matched_entity_ids'] != '').sum():,}")
print(f"  S1 entities with candidates: {(candidate_df['candidate_entity_ids'] != '').sum():,}")
print(f"\nFiles saved to: {OUTPUT_DIR}")
