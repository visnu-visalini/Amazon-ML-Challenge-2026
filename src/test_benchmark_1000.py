import sys
import time
import gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from normalization import (
    normalize_business_name,
    normalize_address,
    normalize_compact,
    normalize_tokens,
    extract_digits,
    normalize_alphanumeric,
)
from similarity_features import calculate_features

BASE_DIR = Path(r"D:\Amazon ML Challenge")
DATASET_DIR = BASE_DIR / "dataset" / "test"
MODEL_FILE = BASE_DIR / "models" / "entity_matcher.txt"
GATE_FILE = BASE_DIR / "models" / "fast_candidate_gate.txt"

S1_FILE = DATASET_DIR / "test_source1.tsv"
S2_FILE = DATASET_DIR / "test_source2.tsv"

GATE_THRESHOLD = 0.10
MAIN_THRESHOLD = 0.80

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

def run_benchmark(num_rows=1000):
    print("=" * 60, flush=True)
    print(f"BENCHMARK WITH {num_rows:,} ROWS", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    print("Loading models...", flush=True)
    main_model = lgb.Booster(model_file=str(MODEL_FILE))
    gate_model = lgb.Booster(model_file=str(GATE_FILE))
    print(f"Models loaded in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    print("Loading S1...", flush=True)
    s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    print(f"S1 loaded ({len(s1):,} rows) in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    print("Pre-normalizing S1...", flush=True)
    s1_ids = s1["entity_id"].tolist()
    s1_countries = [c.strip().lower() for c in s1["country"]]
    
    # Vectorized / fast string operations
    s1_norm_name = [normalize_business_name(x) for x in s1["business_name"]]
    s1_compact_name = [normalize_compact(x) for x in s1["business_name"]]
    s1_tokens_name = [set(normalize_tokens(x)) for x in s1["business_name"]]

    s1_norm_addr = [normalize_address(x) for x in s1["business_address"]]
    s1_compact_addr = [x.replace(" ", "") for x in s1_norm_addr]
    s1_tokens_addr = [set(normalize_tokens(x)) for x in s1["business_address"]]
    s1_digits_addr = [extract_digits(x) for x in s1["business_address"]]
    print(f"S1 pre-normalized in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    print("Building block indexes...", flush=True)
    name_index = {}
    address_index = {}

    for pos in range(len(s1_ids)):
        c = s1_countries[pos]
        nm = s1_norm_name[pos]
        ad = s1_norm_addr[pos]

        if nm:
            nk = (c, nm[:2])
            name_index.setdefault(nk, []).append(pos)

        if ad:
            ak = (c, ad[:12])
            address_index.setdefault(ak, []).append(pos)

    # Convert to numpy arrays for fast lookup
    for k in name_index:
        name_index[k] = np.array(name_index[k], dtype=np.int32)
    for k in address_index:
        address_index[k] = np.array(address_index[k], dtype=np.int32)

    print(f"Indexes built: {len(name_index):,} name blocks, {len(address_index):,} addr blocks in {time.time()-t0:.2f}s", flush=True)

    # Free raw S1 dataframe
    del s1
    gc.collect()

    t0 = time.time()
    print(f"\nLoading {num_rows:,} rows from S2...", flush=True)
    s2 = pd.read_csv(S2_FILE, sep="\t", dtype=str, keep_default_na=False, nrows=num_rows)
    print(f"Loaded in {time.time()-t0:.2f}s", flush=True)

    # Benchmarking candidate generation and scoring
    t_cand_gen = 0.0
    t_fast_feats = 0.0
    t_gate_pred = 0.0
    t_main_feats = 0.0
    t_main_pred = 0.0

    total_candidates = 0
    total_gate_survivors = 0
    total_matches = 0

    t_start_process = time.time()

    # Pre-normalize S2 rows
    s2_ids = s2["entity_id"].tolist()
    s2_countries = [c.strip().lower() for c in s2["country"]]
    s2_norm_name = [normalize_business_name(x) for x in s2["business_name"]]
    s2_compact_name = [normalize_compact(x) for x in s2["business_name"]]
    s2_tokens_name = [set(normalize_tokens(x)) for x in s2["business_name"]]
    s2_norm_addr = [normalize_address(x) for x in s2["business_address"]]
    s2_compact_addr = [x.replace(" ", "") for x in s2_norm_addr]
    s2_tokens_addr = [set(normalize_tokens(x)) for x in s2["business_address"]]
    s2_digits_addr = [extract_digits(x) for x in s2["business_address"]]

    # Buffer for batch gate prediction
    BATCH_SIZE = 50_000
    fast_feature_buf = []
    pair_buf = []  # (s2_idx, s1_pos)

    # Accumulate matches
    matches = []

    def flush_gate():
        nonlocal t_gate_pred, t_main_feats, t_main_pred
        nonlocal total_gate_survivors, total_matches

        if not fast_feature_buf:
            return

        t_g0 = time.time()
        X_fast = np.array(fast_feature_buf, dtype=np.float32)
        gate_probs = gate_model.predict(X_fast)
        t_gate_pred += time.time() - t_g0

        survivors = np.where(gate_probs >= GATE_THRESHOLD)[0]
        total_gate_survivors += len(survivors)

        if len(survivors) > 0:
            t_mf0 = time.time()
            main_rows = []
            survivor_pairs = []
            for s_idx in survivors:
                s2_idx, s1_pos = pair_buf[s_idx]
                survivor_pairs.append((s2_idx, s1_pos))

                # Compute the 2 missing features: name_partial, address_partial
                # Exact calculation from similarity_features.py:
                # name_partial = fuzz.partial_ratio(s1_compact_name[s1_pos], s2_compact_name[s2_idx])
                # address_partial = fuzz.partial_ratio(s1_norm_addr[s1_pos], s2_norm_addr[s2_idx])
                s1_c_nm = s1_compact_name[s1_pos]
                s2_c_nm = s2_compact_name[s2_idx]
                np_val = fuzz.partial_ratio(s1_c_nm, s2_c_nm) if (s1_c_nm and s2_c_nm) else 0.0

                s1_n_ad = s1_norm_addr[s1_pos]
                s2_n_ad = s2_norm_addr[s2_idx]
                ap_val = fuzz.partial_ratio(s1_n_ad, s2_n_ad) if (s1_n_ad and s2_n_ad) else 0.0

                # FAST_FEATURES:
                # 0: name_token_sort, 1: name_token_set, 2: name_character, 3: name_jaccard,
                # 4: address_token_sort, 5: address_token_set, 6: address_character, 7: address_jaccard,
                # 8: address_digit_similarity, 9: country_match
                # MAIN_FEATURES:
                # [name_token_sort, name_token_set, name_character, name_partial, name_jaccard,
                #  address_token_sort, address_token_set, address_character, address_partial,
                #  address_jaccard, address_digit_similarity, country_match]
                f = X_fast[s_idx]
                main_row = [
                    f[0], f[1], f[2], np_val, f[3],
                    f[4], f[5], f[6], ap_val, f[7],
                    f[8], f[9]
                ]
                main_rows.append(main_row)

            X_main = np.array(main_rows, dtype=np.float32)
            t_main_feats += time.time() - t_mf0

            t_mp0 = time.time()
            main_probs = main_model.predict(X_main)
            t_main_pred += time.time() - t_mp0

            match_indices = np.where(main_probs >= MAIN_THRESHOLD)[0]
            total_matches += len(match_indices)
            for m_idx in match_indices:
                s2_idx, s1_pos = survivor_pairs[m_idx]
                matches.append((s1_ids[s1_pos], s2_ids[s2_idx], float(main_probs[m_idx])))

        fast_feature_buf.clear()
        pair_buf.clear()

    for s2_idx in range(len(s2_ids)):
        t_cg0 = time.time()
        c = s2_countries[s2_idx]
        nm = s2_norm_name[s2_idx]
        ad = s2_norm_addr[s2_idx]

        cand_arrs = []
        if nm:
            nk = (c, nm[:2])
            arr = name_index.get(nk)
            if arr is not None:
                cand_arrs.append(arr)
        if ad:
            ak = (c, ad[:12])
            arr = address_index.get(ak)
            if arr is not None:
                cand_arrs.append(arr)

        if not cand_arrs:
            continue

        if len(cand_arrs) == 1:
            candidates = cand_arrs[0]
        else:
            candidates = np.unique(np.concatenate(cand_arrs))

        total_candidates += len(candidates)
        t_cand_gen += time.time() - t_cg0

        # Fast features
        t_ff0 = time.time()
        s2_tok_nm = s2_tokens_name[s2_idx]
        s2_c_nm = s2_compact_name[s2_idx]
        s2_n_nm = nm
        s2_n_ad = ad
        s2_c_ad = s2_compact_addr[s2_idx]
        s2_tok_ad = s2_tokens_addr[s2_idx]
        s2_dig = s2_digits_addr[s2_idx]

        for s1_pos in candidates:
            s1_n_nm = s1_norm_name[s1_pos]
            s1_tok_nm = s1_tokens_name[s1_pos]
            s1_c_nm = s1_compact_name[s1_pos]

            s1_n_ad = s1_norm_addr[s1_pos]
            s1_c_ad = s1_compact_addr[s1_pos]
            s1_tok_ad = s1_tokens_addr[s1_pos]
            s1_dig = s1_digits_addr[s1_pos]

            # Fast 10 features:
            nts = fuzz.token_sort_ratio(s1_n_nm, s2_n_nm) if (s1_n_nm and s2_n_nm) else 0.0
            ntset = fuzz.token_set_ratio(s1_n_nm, s2_n_nm) if (s1_n_nm and s2_n_nm) else 0.0
            nc = fuzz.ratio(s1_c_nm, s2_c_nm) if (s1_c_nm and s2_c_nm) else 0.0
            nj = len(s1_tok_nm & s2_tok_nm) / len(s1_tok_nm | s2_tok_nm) if (s1_tok_nm and s2_tok_nm) else 0.0

            ats = fuzz.token_sort_ratio(s1_n_ad, s2_n_ad) if (s1_n_ad and s2_n_ad) else 0.0
            atset = fuzz.token_set_ratio(s1_n_ad, s2_n_ad) if (s1_n_ad and s2_n_ad) else 0.0
            ac = fuzz.ratio(s1_c_ad, s2_c_ad) if (s1_c_ad and s2_c_ad) else 0.0
            aj = len(s1_tok_ad & s2_tok_ad) / len(s1_tok_ad | s2_tok_ad) if (s1_tok_ad and s2_tok_ad) else 0.0

            if not s1_dig or not s2_dig:
                ds = 0.0
            else:
                ds = fuzz.ratio(s1_dig, s2_dig)

            cm = 1.0  # By hard block on country, countries always match

            fast_feature_buf.append([
                nts, ntset, nc, nj,
                ats, atset, ac, aj,
                ds, cm
            ])
            pair_buf.append((s2_idx, s1_pos))

            if len(fast_feature_buf) >= BATCH_SIZE:
                flush_gate()

        t_fast_feats += time.time() - t_ff0

    flush_gate()

    t_total_process = time.time() - t_start_process
    print("\n" + "=" * 60, flush=True)
    print("BENCHMARK RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"Processed S2 rows          : {num_rows:,}", flush=True)
    print(f"Total raw candidates       : {total_candidates:,} ({total_candidates/num_rows:.1f}/row)", flush=True)
    print(f"Gate survivors (>= {GATE_THRESHOLD:.2f})  : {total_gate_survivors:,} ({total_gate_survivors/max(total_candidates,1)*100:.2f}%)", flush=True)
    print(f"Main matches   (>= {MAIN_THRESHOLD:.2f})  : {total_matches:,}", flush=True)
    print(f"\nTiming breakdown:", flush=True)
    print(f"  Candidate generation     : {t_cand_gen:.2f}s", flush=True)
    print(f"  Fast feature computation : {t_fast_feats:.2f}s", flush=True)
    print(f"  Gate prediction          : {t_gate_pred:.2f}s", flush=True)
    print(f"  Main feature computation : {t_main_feats:.2f}s", flush=True)
    print(f"  Main model prediction    : {t_main_pred:.2f}s", flush=True)
    print(f"  Total processing time    : {t_total_process:.2f}s ({num_rows/t_total_process:.1f} rows/s)", flush=True)

    print(f"\nSample matches found (first 5):", flush=True)
    for m in matches[:5]:
        print(f"  S1: {m[0]} <-> S2: {m[1]} (score: {m[2]:.4f})", flush=True)

if __name__ == "__main__":
    n = 1000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    run_benchmark(n)
