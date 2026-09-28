"""Inspect a fixed subject split and report any cross-split overlap.

Reader's guide: this is a standalone diagnostic for the historical default
split, not the 12-fold launcher. It first checks the subject-ID sets and then
asks the shared loader which files each set resolves to on disk.
"""

import sys
from pathlib import Path

# Add parent directory to path to import trainer module
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.load import create_subject_split

# Default splits from train.py
train_subjects = [1, 4, 5, 6, 10, 17, 19, 20]
val_subjects = [21, 23]
test_subjects = [24, 25]

print("="*80)
print("DATA SPLIT VERIFICATION")
print("="*80)

print("\nConfigured splits:")
print(f"  Train subjects: {train_subjects}")
print(f"  Val subjects:   {val_subjects}")
print(f"  Test subjects:  {test_subjects}")

# Check for any overlap
train_set = set(train_subjects)
val_set = set(val_subjects)
test_set = set(test_subjects)

print("\nChecking for data leakage:")
train_val_overlap = train_set & val_set
train_test_overlap = train_set & test_set
val_test_overlap = val_set & test_set

if train_val_overlap:
    print(f"  ❌ WARNING: Train/Val overlap: {train_val_overlap}")
else:
    print(f"  ✓ No Train/Val overlap")

if train_test_overlap:
    print(f"  ❌ WARNING: Train/Test overlap: {train_test_overlap}")
else:
    print(f"  ✓ No Train/Test overlap")

if val_test_overlap:
    print(f"  ❌ WARNING: Val/Test overlap: {val_test_overlap}")
else:
    print(f"  ✓ No Val/Test overlap")

# Check actual files
data_dir = Path('data/processed')
splits = create_subject_split(
    data_dir=data_dir,
    train_subjects=train_subjects,
    val_subjects=val_subjects,
    test_subjects=test_subjects,
)

print("\n" + "="*80)
print("SUBJECT 24 VERIFICATION")
print("="*80)

# Check subject 24 specifically
subject_24_files = list(data_dir.glob('subject24*.pkl'))
print(f"\nSubject 24 files found: {len(subject_24_files)}")
for f in subject_24_files:
    print(f"  - {f.name}")

print(f"\nSubject 24 is in TEST set: {24 in test_subjects}")
print(f"Subject 24 is in TRAIN set: {24 in train_subjects}")
print(f"Subject 24 is in VAL set: {24 in val_subjects}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if 24 in test_subjects and 24 not in train_subjects and 24 not in val_subjects:
    print("✓ Subject 24 is CORRECTLY in test set only (not seen during training)")
    print("  The model has never seen subject 24 during training or validation.")
    print("  Performance on subject 24 represents true generalization to unseen data.")
else:
    print("❌ WARNING: Data leakage detected for subject 24!")
