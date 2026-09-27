import os
import random
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"D:\Amazon ML Challenge"
TRAINING_DIR = os.path.join(
    BASE_DIR,
    "training_data"
)

S2_FILE = os.path.join(
    TRAINING_DIR,
    "s2_training_features.csv"
)

S3_FILE = os.path.join(
    TRAINING_DIR,
    "s3_training_features.csv"
)

TRAIN_FILE = os.path.join(
    TRAINING_DIR,
    "model_train.csv"
)

VALID_FILE = os.path.join(
    TRAINING_DIR,
    "model_validation.csv"
)

RANDOM_SEED = 42
TRAIN_RATIO = 0.80


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("LOADING FEATURE DATA")
print("=" * 70)

s2 = pd.read_csv(
    S2_FILE
)

s3 = pd.read_csv(
    S3_FILE
)

print(f"S2 rows: {len(s2):,}")
print(f"S3 rows: {len(s3):,}")


# ============================================================
# COMBINE
# ============================================================

print("\nCombining S2 + S3...")

data = pd.concat(
    [s2, s3],
    ignore_index=True
)

print(
    f"Total rows: "
    f"{len(data):,}"
)


# ============================================================
# UNIQUE S1 ENTITIES
# ============================================================

unique_entities = (
    data["source1_entity_id"]
    .unique()
    .tolist()
)

print(
    f"Unique S1 entities: "
    f"{len(unique_entities):,}"
)


# ============================================================
# SHUFFLE ENTITIES
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(
    unique_entities
)

split_index = int(
    len(unique_entities) * TRAIN_RATIO
)

train_entities = set(
    unique_entities[:split_index]
)

validation_entities = set(
    unique_entities[split_index:]
)


# ============================================================
# SPLIT
# ============================================================

train_data = data[
    data["source1_entity_id"].isin(
        train_entities
    )
].copy()

validation_data = data[
    data["source1_entity_id"].isin(
        validation_entities
    )
].copy()


# ============================================================
# SAVE
# ============================================================

train_data.to_csv(
    TRAIN_FILE,
    index=False
)

validation_data.to_csv(
    VALID_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("MODEL DATA PREPARATION COMPLETE")
print("=" * 70)

print(
    f"Training rows: "
    f"{len(train_data):,}"
)

print(
    f"Validation rows: "
    f"{len(validation_data):,}"
)

print(
    f"\nTraining positives: "
    f"{(train_data['label'] == 1).sum():,}"
)

print(
    f"Training negatives: "
    f"{(train_data['label'] == 0).sum():,}"
)

print(
    f"\nValidation positives: "
    f"{(validation_data['label'] == 1).sum():,}"
)

print(
    f"Validation negatives: "
    f"{(validation_data['label'] == 0).sum():,}"
)

print(
    f"\nTraining file:\n{TRAIN_FILE}"
)

print(
    f"\nValidation file:\n{VALID_FILE}"
)
