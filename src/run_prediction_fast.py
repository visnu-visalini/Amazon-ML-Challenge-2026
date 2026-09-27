import gc
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from similarity_features import calculate_features

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
FEATURE_BATCH_SIZE = 20000


def norm(v):
    if not v or v != v:
        return ""
    s = str(v).lower().strip()
    out = []
    space = False

    for ch in s:
        if ch.isalnum():
            out.append(ch)
            space = False
        elif not space:
            out.append(" ")
            space = True

    return "".join(out).strip()


def name_key(country, name):
    n = norm(name).replace(" ", "")
    if not n:
        return None
    return f"{str(country).lower().strip()}|{n[:3]}"


def address_key(country, address):
    a = norm(address)
    if not a:
        return None
    return f"{str(country).lower().strip()}|{a[:12]}"


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


print("=" * 70)
print("FAST PRODUCTION PREDICTION")
print("=" * 70)

print("\nLoading models...")
main_model = lgb.Booster(model_file=str(MODEL_FILE))
gate_model = lgb.Booster(model_file=str(GATE_FILE))
print("Models loaded.")

print("\nLoading S1...")
s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s1_ids = s1["entity_id"].tolist()
s1_names = s1["business_name"].tolist()
s1_addresses = s1["business_address"].tolist()
s1_countries = s1["country"].tolist()

print(f"S1 rows: {len(s1_ids):,}")

print("\nBuilding indexes...")

name_idx = defaultdict(list)
address_idx = defaultdict(list)

for pos in range(len(s1_ids)):
    nk = name_key(s1_countries[pos], s1_names[pos])
    if nk:
        name_idx[nk].append(pos)

    ak = address_key(s1_countries[pos], s1_addresses[pos])
    if ak:
        address_idx[ak].append(pos)

print(f"Name3 blocks: {len(name_idx):,}")
print(f"Address12 blocks: {len(address_idx):,}")

matches = defaultdict(set)
candidates = defaultdict(set)


def process_source(path, label):

    print("\n" + "=" * 70)
    print(f"PROCESSING {label}")
    print("=" * 70)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(f"Rows: {len(df):,}")

    feature_buf = []
    pair_buf = []

    total_candidates = 0
    gate_pass = 0
    total_matches = 0
    processed = 0

    def flush():

        nonlocal gate_pass, total_matches

        if not feature_buf:
            return

        X = pd.DataFrame(feature_buf)

        gate_probs = gate_model.predict(X[GATE_FEATURES])

        survivors = [
            i for i, p in enumerate(gate_probs)
            if p >= GATE_THRESHOLD
        ]

        gate_pass += len(survivors)

        if survivors:

            X_main = X.iloc[survivors][MAIN_FEATURES]
            probs = main_model.predict(X_main)

            for j, prob in enumerate(probs):

                idx = survivors[j]
                src_id, s1_pos = pair_buf[idx]
                s1_id = s1_ids[s1_pos]

                candidates[s1_id].add(src_id)

                if prob >= MAIN_THRESHOLD:
                    matches[s1_id].add(src_id)
                    total_matches += 1

        feature_buf.clear()
        pair_buf.clear()

        del X
        del gate_probs
        gc.collect()

    for row in df.itertuples(index=False):

        src_id = row.entity_id
        src_name = row.business_name
        src_address = row.business_address
        src_country = row.country

        positions = set()

        nk = name_key(src_country, src_name)

        if nk:
            positions.update(name_idx.get(nk, []))

        ak = address_key(src_country, src_address)

        if ak:
            positions.update(address_idx.get(ak, []))

        total_candidates += len(positions)

        for s1_pos in positions:

            feats = calculate_features(
                s1_names[s1_pos],
                s1_addresses[s1_pos],
                s1_countries[s1_pos],
                src_name,
                src_address,
                src_country
            )

            feature_buf.append(feats)
            pair_buf.append((src_id, s1_pos))

            if len(feature_buf) >= FEATURE_BATCH_SIZE:
                flush()

        processed += 1

        if processed % 100000 == 0:

            print(
                f"{label}: {processed:,}/{len(df):,} | "
                f"candidates={total_candidates:,} | "
                f"gate={gate_pass:,} | "
                f"matches={total_matches:,}",
                flush=True
            )

    flush()

    print(f"\n{label} COMPLETE")
    print(f"Candidates : {total_candidates:,}")
    print(f"Gate pass  : {gate_pass:,}")
    print(f"Matches    : {total_matches:,}")


process_source(S2_FILE, "S2")

print("\nWriting S2 checkpoint...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for s1_id in s1_ids:
    rows.append({
        "source1_entity_id": s1_id,
        "matched_entity_ids": ",".join(
            sorted(matches.get(s1_id, set()))
        )
    })

pd.DataFrame(rows).to_csv(
    OUTPUT_DIR / "s2_fast_partial.tsv",
    sep="\t",
    index=False
)

del rows
gc.collect()

print("S2 checkpoint saved.")

process_source(S3_FILE, "S3")

print("\n" + "=" * 70)
print("BUILDING FINAL SUBMISSION")
print("=" * 70)

matching_rows = []
candidate_rows = []

for s1_id in s1_ids:

    m = matches.get(s1_id, set())
    c = candidates.get(s1_id, set()) | m

    matching_rows.append({
        "source1_entity_id": s1_id,
        "matched_entity_ids": ",".join(sorted(m))
    })

    candidate_rows.append({
        "source1_entity_id": s1_id,
        "candidate_entity_ids": ",".join(sorted(c))
    })

pd.DataFrame(matching_rows).to_csv(
    OUTPUT_DIR / "matching_results.tsv",
    sep="\t",
    index=False
)

pd.DataFrame(candidate_rows).to_csv(
    OUTPUT_DIR / "candidate_pairs.tsv",
    sep="\t",
    index=False
)

print("\nDONE")
print(f"matching_results.tsv rows: {len(matching_rows):,}")
print(f"candidate_pairs.tsv rows : {len(candidate_rows):,}")
print(f"Output directory: {OUTPUT_DIR}")