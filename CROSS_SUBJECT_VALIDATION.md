# Cross-Subject Validation

SonoRank is evaluated with 12-fold leave-one-subject-out validation. Every subject is used as the test subject once.

## Subjects

The subject order is:

```text
1, 4, 5, 6, 10, 17, 19, 20, 21, 23, 24, 25
```

For each fold:

- One subject is held out for testing.
- The preceding subject in the list is used for validation.
- The remaining ten subjects are used for training.
- The list wraps around, so subject 25 is the validation subject when subject 1 is tested.

Examples:

```text
Test 1: validation 25, training 4 5 6 10 17 19 20 21 23 24
Test 4: validation 1, training 5 6 10 17 19 20 21 23 24 25
Test 5: validation 4, training 1 6 10 17 19 20 21 23 24 25
```

All partitions are subject-disjoint. Frames or sessions from a test subject must not appear in training or validation.

## Training order

The SonoRank stages are trained in this order:

```text
Level 1: scripts/run_cross_subject_embedding.sh
Level 2: scripts/run_cross_subject_multi_finger_ranking.sh
Level 3: scripts/run_cross_subject_flex_embedding.sh
```

Level 2 loads the encoder checkpoint from Level 1 for the same fold. Level 3 loads the ranking checkpoint from Level 2 for that fold.

## Before running a launcher

The launchers retain the machine settings and fold subsets used for individual experiment batches. Check these variables before execution:

- `PYTHON_BIN` selects the Python interpreter.
- `SELECTED_TEST_SUBJECTS` selects the folds to run.
- `GPU_IDS` assigns each selected fold to a GPU.

Use all twelve subject IDs in `SELECTED_TEST_SUBJECTS` for a complete validation run. The number of entries in `GPU_IDS` must match the number of selected subjects.

See [scripts/README.md](scripts/README.md) for launcher details.

## Outputs

Each fold writes to its own experiment directory. The directory normally contains:

- `config.json` with the saved command-line settings
- `checkpoint_best.pth` selected on the validation subject
- any periodic or final checkpoints written by that trainer
- validation and test metrics produced by the selected trainer
- logs for that fold

Report aggregate results across all twelve held-out test subjects. Keep the fold-level results so subject variability remains visible.
