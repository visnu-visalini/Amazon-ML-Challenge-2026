import sys
import time
import gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent))
from normalization import (
    normalize_business_name,
    normalize_address,
    normalize_compact,
    normalize_tokens,
    extract_digits,
)

BASE_DIR = Path(r"D:\Amazon ML Challenge")
DATASET_DIR = BASE_DIR / "dataset" / "test"
MODEL_FILE = BASE_DIR / "models" / "entity_matcher.txt"
GATE_FILE = BASE_DIR / "models" / "fast_candidate_gate.txt"

S1_FILE = DATASET_DIR / "test_source1.tsv"
S2_FILE = DATASET_DIR / "test_source2.tsv"

GATE_THRESHOLD = 0.10
MAIN_THRESHOLD = 0.80
NAME_PREFILTER_SCORE = 35.0
BATCH_SIZE = 10_000

def test_pipeline(num_rows=1000):
    print("=" * 65, flush=True)
    print(f"PIPELINE TEST: {num_rows:,} S2 ROWS", flush=True)
    print("=" * 65, flush=True)

    t0 = time.time()
    main_model = lgb.Booster(model_file=str(MODEL_FILE))
    gate_model = lgb.Booster(model_file=str(GATE_FILE))
    print(f"Models loaded in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    print(f"S1 loaded ({len(s1):,} rows) in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    s1_ids = s1["entity_id"].tolist()
    s1_countries = [c.strip().lower() for c in s1["country"]]
    s1_norm_name = [normalize_business_name(x) for x in s1["business_name"]]
    s1_compact_name = [normalize_compact(x) for x in s1["business_name"]]
    s1_tokens_name = [set(normalize_tokens(x)) for x in s1["business_name"]]
    s1_norm_addr = [normalize_address(x) for x in s1["business_address"]]
    s1_compact_addr = [x.replace(" ", "") for x in s1_norm_addr]
    s1_tokens_addr = [set(normalize_tokens(x)) for x in s1["business_address"]]
    s1_digits_addr = [extract_digits(x) for x in s1["business_address"]]
    print(f"S1 pre-normalized in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    # name_index: key -> (list_of_compact_names, list_of_positions)
    name_index = {}
    address_index = {}

    for pos in range(len(s1_ids)):
        c = s1_countries[pos]
        nm = s1_norm_name[pos]
        ad = s1_norm_addr[pos]

        if nm:
            nk = (c, nm[:2])
            if nk not in name_index:
                name_index[nk] = ([], [])
            name_index[nk][0].append(s1_compact_name[pos])
            name_index[nk][1].append(pos)

        if ad:
            ak = (c, ad[:12])
            address_index.setdefault(ak, []).append(pos)

    for k in address_index:
        address_index[k] = np.array(address_index[k], dtype=np.int32)

    print(f"Indexes built: {len(name_index):,} name blocks, {len(address_index):,} addr blocks in {time.time()-t0:.2f}s", flush=True)

    del s1
    gc.collect()

    s2 = pd.read_csv(S2_FILE, sep="\t", dtype=str, keep_default_na=False, nrows=num_rows)
    s2_ids = s2["entity_id"].tolist()
    s2_countries = [c.strip().lower() for c in s2["country"]]
    s2_norm_name = [normalize_business_name(x) for x in s2["business_name"]]
    s2_compact_name = [normalize_compact(x) for x in s2["business_name"]]
    s2_tokens_name = [set(normalize_tokens(x)) for x in s2["business_name"]]
    s2_norm_addr = [normalize_address(x) for x in s2["business_address"]]
    s2_compact_addr = [x.replace(" ", "") for x in s2_norm_addr]
    s2_tokens_addr = [set(normalize_tokens(x)) for x in s2["business_address"]]
    s2_digits_addr = [extract_digits(x) for x in s2["business_address"]]

    print(f"\nProcessing {num_rows:,} S2 rows...", flush=True)
    t_start = time.time()

    total_candidates = 0
    total_gate_survivors = 0
    total_matches = 0

    fast_feature_buf = []
    pair_buf = []
    matches = []

    def flush_batch():
        nonlocal total_gate_survivors, total_matches

        if not fast_feature_buf:
            return

        X_fast = np.array(fast_feature_buf, dtype=np.float32)
        gate_probs = gate_model.predict(X_fast)
        survivors = np.where(gate_probs >= GATE_THRESHOLD)[0]
        total_gate_survivors += len(survivors)

        if len(survivors) > 0:
            main_rows = []
            surv_pairs = []
            for s_idx in survivors:
                s2_idx, s1_pos = pair_buf[s_idx]
                surv_pairs.append((s2_idx, s1_pos))

                s1_c_nm = s1_compact_name[s1_pos]
                s2_c_nm = s2_compact_name[s2_idx]
                np_val = fuzz.partial_ratio(s1_c_nm, s2_c_nm) if (s1_c_nm and s2_c_nm) else 0.0

                s1_n_ad = s1_norm_addr[s1_pos]
                s2_n_ad = s2_norm_addr[s2_idx]
                ap_val = fuzz.partial_ratio(s1_n_ad, s2_n_ad) if (s1_n_ad and s2_n_ad) else 0.0

                f = X_fast[s_idx]
                main_row = [
                    f[0], f[1], f[2], np_val, f[3],
                    f[4], f[5], f[6], ap_val, f[7],
                    f[8], f[9]
                ]
                main_rows.append(main_row)

            X_main = np.array(main_rows, dtype=np.float32)
            main_probs = main_model.predict(X_main)

            match_indices = np.where(main_probs >= MAIN_THRESHOLD)[0]
            total_matches += len(match_indices)
            for m_idx in match_indices:
                s2_idx, s1_pos = surv_pairs[m_idx]
                matches.append((s1_ids[s1_pos], s2_ids[s2_idx], float(main_probs[m_idx])))

        fast_feature_buf.clear()
        pair_buf.clear()

    for s2_idx in range(len(s2_ids)):
        c = s2_countries[s2_idx]
        nm = s2_norm_name[s2_idx]
        c_nm = s2_compact_name[s2_idx]
        ad = s2_norm_addr[s2_idx]

        cand_set = set()

        # Address12 candidates bypass name filter
        if ad and (c, ad[:12]) in address_index:
            cand_set.update(address_index[(c, ad[:12])])

        # Name2 candidates with C++ RapidFuzz filter
        if nm and c_nm and (c, nm[:2]) in name_index:
            bucket_names, bucket_pos = name_index[(c, nm[:2])]
            filtered = process.extract(c_nm, bucket_names, scorer=fuzz.ratio, score_cutoff=NAME_PREFILTER_SCORE, limit=None)
            for _, score, local_idx in filtered:
                cand_set.add(bucket_pos[local_idx])

        if not cand_set:
            continue

        total_candidates += len(cand_set)

        s2_tok_nm = s2_tokens_name[s2_idx]
        s2_c_nm = c_nm
        s2_n_nm = nm
        s2_n_ad = ad
        s2_c_ad = s2_compact_addr[s2_idx]
        s2_tok_ad = s2_tokens_addr[s2_idx]
        s2_dig = s2_digits_addr[s2_idx]

        for s1_pos in cand_set:
            s1_n_nm = s1_norm_name[s1_pos]
            s1_tok_nm = s1_tokens_name[s1_pos]
            s1_c_nm = s1_compact_name[s1_pos]
            s1_n_ad = s1_norm_addr[s1_pos]
            s1_c_ad = s1_compact_addr[s1_pos]
            s1_tok_ad = s1_tokens_addr[s1_pos]
            s1_dig = s1_digits_addr[s1_pos]

            nts = fuzz.token_sort_ratio(s1_n_nm, s2_n_nm) if (s1_n_nm and s2_n_nm) else 0.0
            ntset = fuzz.token_set_ratio(s1_n_nm, s2_n_nm) if (s1_n_nm and s2_n_nm) else 0.0
            nc = fuzz.ratio(s1_c_nm, s2_c_nm) if (s1_c_nm and s2_c_nm) else 0.0
            nj = len(s1_tok_nm & s2_tok_nm) / len(s1_tok_nm | s2_tok_nm) if (s1_tok_nm and s2_tok_nm) else 0.0

            ats = fuzz.token_sort_ratio(s1_n_ad, s2_n_ad) if (s1_n_ad and s2_n_ad) else 0.0
            atset = fuzz.token_set_ratio(s1_n_ad, s2_n_ad) if (s1_n_ad and s2_n_ad) else 0.0
            ac = fuzz.ratio(s1_c_ad, s2_c_ad) if (s1_c_ad and s2_c_ad) else 0.0
            aj = len(s1_tok_ad & s2_tok_ad) / len(s1_tok_ad | s2_tok_ad) if (s1_tok_ad and s2_tok_ad) else 0.0

            ds = fuzz.ratio(s1_dig, s2_dig) if (s1_dig and s2_dig) else 0.0
            cm = 1.0

            fast_feature_buf.append([
                nts, ntset, nc, nj,
                ats, atset, ac, aj,
                ds, cm
            ])
            pair_buf.append((s2_idx, s1_pos))

            if len(fast_feature_buf) >= BATCH_SIZE:
                flush_batch()

        if (s2_idx + 1) % 200 == 0:
            dt = time.time() - t_start
            print(f"  Processed {s2_idx+1:,}/{num_rows:,} | Candidates: {total_candidates:,} | Gate pass: {total_gate_survivors:,} | Matches: {total_matches:,} ({dt:.1f}s)", flush=True)

    flush_batch()
    total_time = time.time() - t_start

    print("\n" + "=" * 65, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Processed S2 rows          : {num_rows:,}", flush=True)
    print(f"Total candidates           : {total_candidates:,} ({total_candidates/num_rows:.1f}/row)", flush=True)
    print(f"Gate survivors (>= {GATE_THRESHOLD:.2f})  : {total_gate_survivors:,} ({total_gate_survivors/max(total_candidates,1)*100:.2f}%)", flush=True)
    print(f"Final matches  (>= {MAIN_THRESHOLD:.2f})  : {total_matches:,} ({total_matches/num_rows:.3f}/row)", flush=True)
    print(f"Processing time            : {total_time:.2f}s ({num_rows/total_time:.1f} rows/s)", flush=True)
    print(f"Sample matches:", flush=True)
    for m in matches[:5]:
        print(f"  {m[0]} <-> {m[1]} (prob: {m[2]:.4f})", flush=True)

if __name__ == "__main__":
    n = 1000
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    test_pipeline(n)
