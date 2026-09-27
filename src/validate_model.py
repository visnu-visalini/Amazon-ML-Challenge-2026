import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, fbeta_score

BASE_DIR = r"D:\Amazon ML Challenge"

VALID_FILE = os.path.join(
    BASE_DIR, "training_data", "model_validation.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR, "models", "entity_matcher.txt"
)

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

print("=" * 60)
print("LOADING VALIDATION DATA + MODEL")
print("=" * 60)

df = pd.read_csv(VALID_FILE)

model = lgb.Booster(model_file=MODEL_FILE)

prob = model.predict(df[FEATURES])
df["probability"] = prob

y = df["label"]

print(f"Validation pairs   : {len(df):,}")
print(f"Validation entities: {df['source1_entity_id'].nunique():,}")


# ============================================================
# FINE THRESHOLD SEARCH
# ============================================================

print("\n" + "=" * 60)
print("FINE THRESHOLD SEARCH")
print("=" * 60)

best_threshold = 0
best_f05 = -1

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F0.5':<12}"
)

for i in range(60, 96):

    threshold = i / 100

    predictions = (
        df["probability"] >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y,
        predictions,
        beta=0.5,
        zero_division=0
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f05:<12.4f}"
    )

    if f05 > best_f05:
        best_f05 = f05
        best_threshold = threshold


print("\nBest threshold:", f"{best_threshold:.2f}")
print("Best F0.5     :", f"{best_f05:.6f}")


# ============================================================
# MULTI-MATCH ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("MULTI-MATCH ANALYSIS")
print("=" * 60)

positive = df[df["label"] == 1].copy()

positive["predicted"] = (
    positive["probability"] >= best_threshold
).astype(int)

true_matches_per_entity = (
    positive
    .groupby("source1_entity_id")
    .size()
)

retained_matches_per_entity = (
    positive
    .groupby("source1_entity_id")["predicted"]
    .sum()
)

multi_match_entities = (
    true_matches_per_entity > 1
).sum()

all_matches_retained = (
    retained_matches_per_entity
    == true_matches_per_entity
).sum()

print(
    f"Entities with >=1 known match : "
    f"{len(true_matches_per_entity):,}"
)

print(
    f"Entities with multiple matches: "
    f"{multi_match_entities:,}"
)

print(
    f"Entities retaining ALL matches: "
    f"{all_matches_retained:,}"
)

print(
    "All-match retention rate       : "
    f"{all_matches_retained / len(true_matches_per_entity):.4f}"
)


# ============================================================
# FALSE POSITIVE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FALSE POSITIVE ANALYSIS")
print("=" * 60)

df["predicted"] = (
    df["probability"] >= best_threshold
).astype(int)

false_positives = df[
    (df["predicted"] == 1) &
    (df["label"] == 0)
]

fp_per_entity = (
    false_positives
    .groupby("source1_entity_id")
    .size()
)

print(
    f"Entities with >=1 false positive: "
    f"{len(fp_per_entity):,}"
)

if len(fp_per_entity) > 0:

    print(
        f"Maximum false positives/entity : "
        f"{fp_per_entity.max():,}"
    )

    print(
        f"Mean false positives/entity    : "
        f"{fp_per_entity.mean():.3f}"
    )


# ============================================================
# SAVE SCORED VALIDATION DATA
# ============================================================

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "training_data",
    "validation_scored.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("VALIDATION SCORING COMPLETE")
print("=" * 60)

print("Saved:")
print(OUTPUT_FILE)