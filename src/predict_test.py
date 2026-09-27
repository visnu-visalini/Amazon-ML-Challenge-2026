"""
Production prediction pipeline for Amazon ML Challenge 2026.
Generates candidate matches from Source 2 and Source 3 for Source 1.

Pipeline:
1. Country hard blocking + name2 / address12 candidate generation.
2. Fast Candidate Gate (LightGBM, 10 fast features) at threshold >= 0.10.
3. Full 12-feature calculation on gate survivors.
4. Main Entity Matcher (LightGBM, 12 features) at threshold >= 0.80.
5. Preserves all qualifying one-to-many matches.
6. Memory-safe streaming batch architecture.
"""

import gc
import os
import sys
import time
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

GATE_THRESHOLD = 0.10
MAIN_THRESHOLD = 0.80
NAME_PREFILTER_SCORE = 35.0

# Batch size for model inference
BATCH_SIZE = 25_000

# ============================================================
# LOAD MODELS & S1
# ============================================================

def init_pipeline():
    print("=" * 65, flush=True)
    print("INITIALIZING PREDICTION PIPELINE", flush=True)
    print("=" * 65, flush=True)

    t0 = time.time()
    main_model = lgb.Booster(model_file=str(MODEL_FILE))
    gate_model = lgb.Booster(model_file=str(GATE_FILE))
    print(f"Models loaded in {time.time()-t0:.2f}s", flush=True)
    print(f"  Fast gate threshold : {GATE_THRESHOLD:.2f}", flush=True)
    print(f"  Main model threshold: {MAIN_THRESHOLD:.2f}", flush=True)

    t0 = time.time()
    print("\nLoading test_source1.tsv...", flush=True)
    s1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, keep_default_na=False)
    s1_rows = len(s1)
    print(f"Loaded {s1_rows:,} S1 rows in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    print("Pre-normalizing S1 representations...", flush=True)
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
    print("Building blocking indexes (name2 & address12)...", flush=True)
    name_index = {}
    address_index = {}

    for pos in range(s1_rows):
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

    s1_data = {
        "ids": s1_ids,
        "countries": s1_countries,
        "norm_name": s1_norm_name,
        "compact_name": s1_compact_name,
        "tokens_name": s1_tokens_name,
        "norm_addr": s1_norm_addr,
        "compact_addr": s1_compact_addr,
        "tokens_addr": s1_tokens_addr,
        "digits_addr": s1_digits_addr,
        "name_index": name_index,
        "address_index": address_index,
    }

    return main_model, gate_model, s1_data

# ============================================================
# SOURCE PROCESSOR
# ============================================================

def process_source(
    source_file,
    source_name,
    output_file,
    main_model,
    gate_model,
    s1_data,
    max_rows=None,
    chunk_size=100_000,
):
    print("\n" + "=" * 65, flush=True)
    print(f"PROCESSING {source_name}: {source_file.name}", flush=True)
    if max_rows:
        print(f"  Row limit: {max_rows:,}", flush=True)
    print(f"  Output   : {output_file}", flush=True)
    print("=" * 65, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    s1_ids = s1_data["ids"]
    s1_norm_name = s1_data["norm_name"]
    s1_compact_name = s1_data["compact_name"]
    s1_tokens_name = s1_data["tokens_name"]
    s1_norm_addr = s1_data["norm_addr"]
    s1_compact_addr = s1_data["compact_addr"]
    s1_tokens_addr = s1_data["tokens_addr"]
    s1_digits_addr = s1_data["digits_addr"]
    name_index = s1_data["name_index"]
    address_index = s1_data["address_index"]

    total_source_rows = 0
    total_raw_candidates = 0
    total_gate_survivors = 0
    total_matches = 0

    output_header_written = False

    fast_feature_buf = []
    pair_buf = []  # (src_id, s1_pos, src_compact_name, src_norm_addr)

    def flush_batch():
        nonlocal total_gate_survivors, total_matches, output_header_written

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
                src_id, s1_pos, src_c_nm, src_n_ad = pair_buf[s_idx]
                surv_pairs.append((src_id, s1_pos))

                s1_c_nm = s1_compact_name[s1_pos]
                np_val = fuzz.partial_ratio(s1_c_nm, src_c_nm) if (s1_c_nm and src_c_nm) else 0.0

                s1_n_ad = s1_norm_addr[s1_pos]
                ap_val = fuzz.partial_ratio(s1_n_ad, src_n_ad) if (s1_n_ad and src_n_ad) else 0.0

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
            if len(match_indices) > 0:
                total_matches += len(match_indices)
                match_records = []
                for m_idx in match_indices:
                    src_id, s1_pos = surv_pairs[m_idx]
                    match_records.append({
                        "source1_entity_id": s1_ids[s1_pos],
                        "matched_entity_id": src_id,
                        "score": float(main_probs[m_idx]),
                    })

                df_matches = pd.DataFrame(match_records)
                df_matches.to_csv(
                    output_file,
                    mode="a",
                    sep=",",
                    header=not output_header_written,
                    index=False,
                )
                output_header_written = True

        fast_feature_buf.clear()
        pair_buf.clear()

    t_start = time.time()
    reader = pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )

    for chunk in reader:
        if max_rows and total_source_rows >= max_rows:
            break

        if max_rows and total_source_rows + len(chunk) > max_rows:
            chunk = chunk.iloc[:max_rows - total_source_rows]

        src_ids = chunk["entity_id"].tolist()
        src_countries = [c.strip().lower() for c in chunk["country"]]
        src_norm_name = [normalize_business_name(x) for x in chunk["business_name"]]
        src_compact_name = [normalize_compact(x) for x in chunk["business_name"]]
        src_tokens_name = [set(normalize_tokens(x)) for x in chunk["business_name"]]

        src_norm_addr = [normalize_address(x) for x in chunk["business_address"]]
        src_compact_addr = [x.replace(" ", "") for x in src_norm_addr]
        src_tokens_addr = [set(normalize_tokens(x)) for x in chunk["business_address"]]
        src_digits_addr = [extract_digits(x) for x in chunk["business_address"]]

        for i in range(len(src_ids)):
            c = src_countries[i]
            nm = src_norm_name[i]
            c_nm = src_compact_name[i]
            ad = src_norm_addr[i]
            sid = src_ids[i]

            cand_set = set()
            cand_nc = {}

            # Address12 candidates bypass name pre-filter
            if ad and (c, ad[:12]) in address_index:
                cand_set.update(address_index[(c, ad[:12])])

            # Name2 candidates with C++ RapidFuzz pre-filter
            if nm and c_nm and (c, nm[:2]) in name_index:
                bucket_names, bucket_pos = name_index[(c, nm[:2])]
                filtered = process.extract(c_nm, bucket_names, scorer=fuzz.ratio, score_cutoff=NAME_PREFILTER_SCORE, limit=None)
                for _, score, local_idx in filtered:
                    p = bucket_pos[local_idx]
                    cand_set.add(p)
                    cand_nc[p] = score

            if not cand_set:
                total_source_rows += 1
                continue

            total_raw_candidates += len(cand_set)

            s2_tok_nm = src_tokens_name[i]
            s2_c_nm = c_nm
            s2_n_nm = nm
            s2_n_ad = ad
            s2_c_ad = src_compact_addr[i]
            s2_tok_ad = src_tokens_addr[i]
            s2_dig = src_digits_addr[i]

            for s1_pos in cand_set:
                s1_n_nm = s1_norm_name[s1_pos]
                s1_tok_nm = s1_tokens_name[s1_pos]
                s1_c_nm = s1_compact_name[s1_pos]
                s1_n_ad = s1_norm_addr[s1_pos]
                s1_c_ad = s1_compact_addr[s1_pos]
                s1_tok_ad = s1_tokens_addr[s1_pos]
                s1_dig = s1_digits_addr[s1_pos]

                # Fast name features with fast-path equality checks
                if s1_n_nm == s2_n_nm:
                    nts = 100.0 if s1_n_nm else 0.0
                    ntset = 100.0 if s1_n_nm else 0.0
                    nc = 100.0 if s1_n_nm else 0.0
                    nj = 1.0 if s1_n_nm else 0.0
                else:
                    nts = fuzz.token_sort_ratio(s1_n_nm, s2_n_nm) if (s1_n_nm and s2_n_nm) else 0.0
                    ntset = fuzz.token_set_ratio(s1_n_nm, s2_n_nm) if (s1_n_nm and s2_n_nm) else 0.0
                    nc = cand_nc.get(s1_pos)
                    if nc is None:
                        nc = fuzz.ratio(s1_c_nm, s2_c_nm) if (s1_c_nm and s2_c_nm) else 0.0
                    nj = len(s1_tok_nm & s2_tok_nm) / len(s1_tok_nm | s2_tok_nm) if (s1_tok_nm and s2_tok_nm) else 0.0

                # Fast address features with fast-path equality checks
                if s1_n_ad == s2_n_ad:
                    ats = 100.0 if s1_n_ad else 0.0
                    atset = 100.0 if s1_n_ad else 0.0
                    ac = 100.0 if s1_n_ad else 0.0
                    aj = 1.0 if s1_n_ad else 0.0
                else:
                    ats = fuzz.token_sort_ratio(s1_n_ad, s2_n_ad) if (s1_n_ad and s2_n_ad) else 0.0
                    atset = fuzz.token_set_ratio(s1_n_ad, s2_n_ad) if (s1_n_ad and s2_n_ad) else 0.0
                    ac = fuzz.ratio(s1_c_ad, s2_c_ad) if (s1_c_ad and s2_c_ad) else 0.0
                    aj = len(s1_tok_ad & s2_tok_ad) / len(s1_tok_ad | s2_tok_ad) if (s1_tok_ad and s2_tok_ad) else 0.0

                # Digits
                if s1_dig == s2_dig:
                    ds = 100.0 if s1_dig else 0.0
                else:
                    ds = fuzz.ratio(s1_dig, s2_dig) if (s1_dig and s2_dig) else 0.0

                cm = 1.0

                fast_feature_buf.append([
                    nts, ntset, nc, nj,
                    ats, atset, ac, aj,
                    ds, cm
                ])
                pair_buf.append((sid, s1_pos, s2_c_nm, s2_n_ad))

                if len(fast_feature_buf) >= BATCH_SIZE:
                    flush_batch()

            total_source_rows += 1
            if total_source_rows % 1_000 == 0:
                elapsed = time.time() - t_start
                rate = total_source_rows / max(elapsed, 0.01)
                print(
                    f"  {source_name} rows: {total_source_rows:,} | "
                    f"Candidates: {total_raw_candidates:,} | "
                    f"Gate survivors: {total_gate_survivors:,} | "
                    f"Matches: {total_matches:,} | "
                    f"Rate: {rate:.1f} rows/s ({elapsed:.1f}s)",
                    flush=True
                )

    flush_batch()
    total_elapsed = time.time() - t_start

    print("\n" + "=" * 65, flush=True)
    print(f"{source_name} COMPLETED IN {total_elapsed:.2f}s", flush=True)
    print("=" * 65, flush=True)
    print(f"Total {source_name} rows processed : {total_source_rows:,}", flush=True)
    print(f"Total raw candidates generated     : {total_raw_candidates:,}", flush=True)
    print(f"Gate survivors (>= {GATE_THRESHOLD:.2f})           : {total_gate_survivors:,} ({total_gate_survivors/max(total_raw_candidates,1)*100:.2f}%)", flush=True)
    print(f"Final matches  (>= {MAIN_THRESHOLD:.2f})           : {total_matches:,}", flush=True)
    print(f"Average throughput                 : {total_source_rows/max(total_elapsed, 0.01):.1f} rows/s", flush=True)
    print(f"Output saved to                    : {output_file}", flush=True)

    return total_source_rows, total_matches, total_elapsed

if __name__ == "__main__":
    main_model, gate_model, s1_data = init_pipeline()
    
    # Check command-line arguments for test limits
    limit = 1000
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "full":
            limit = None
        else:
            limit = int(sys.argv[1])

    s2_out = OUTPUT_DIR / "s2_matches_test.csv"
    process_source(
        S2_FILE, "S2", s2_out,
        main_model, gate_model, s1_data,
        max_rows=limit
    )