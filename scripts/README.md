# Scripts

Shell scripts for running cross-subject validation and hyperparameter sweeps across multiple GPUs.

## Required local configuration

The launchers contain machine-specific settings from the original experiments.
**Do not run them unchanged on another system.** Check these variables near the
top of the selected script:

- `PYTHON_BIN`: replace the author's absolute interpreter path with the Python
  executable from your SonoRank environment.
- `SELECTED_TEST_SUBJECTS`: several scripts retain only the fold subset used in
  one original batch. Use all twelve subject IDs for a complete evaluation.
- `GPU_IDS`: assign each selected fold to GPU indices that exist on your
  machine.
- `METHODS`: in the classic-baseline launcher, select every baseline that you
  intend to run.
- `--data-dir`: some trainer defaults still point to the author's original
  filesystem. Pass `data/processed_h5/h5_files` explicitly or add it to the
  command assembled by the launcher.

The scripts do not detect available GPUs or choose fold batches automatically.

## Cross-subject validation

The scripts support leave-one-subject-out training for all 12 subjects (IDs: 1, 4, 5, 6, 10, 17, 19, 20, 21, 23, 24, 25). The checked-in subject and GPU arrays reflect the original batched runs, so configure them as described above before execution.

Training order matters because of the dependency chain:

```
1. run_cross_subject_embedding.sh           (Level 1, no dependencies)
2. run_cross_subject_multi_finger_ranking.sh (Level 2, requires Level 1 checkpoints)
3. run_cross_subject_flex_embedding.sh       (Level 3, requires Level 2 checkpoints)
```

Baselines have no dependencies on each other:

```
run_cross_subject_cbmf.sh                   (CBMF, Zadok et al. 2025)
run_cross_subject_bimbraw.sh                (BimbrawVGG, Bimbraw et al. 2023)
run_cross_subject_classic_baselines.sh      (Castellini / Huang / McIntosh, CPU only)
```

### Customizing folds

Each script defines a `SELECTED_TEST_SUBJECTS` array. To run a subset of folds:

```bash
SELECTED_TEST_SUBJECTS=(4 10 17 23)  # only these 4 subjects as test
```

### Circular validation assignment

For each test subject N, the validation subject is the previous subject in the sorted list (wrapping around). The remaining 10 subjects are used for training.

## Ablation sweeps

| Script | Ablation | Paper figure |
|--------|----------|-------------|
| `sweep_sequence_length.sh` | Sequence length (2, 4, 8, 16 frames) | Fig. 5a |
| `sweep_subject_count.sh` | Training set size (1-9 subjects) | Fig. 5b |
| `sweep_uncertainty_weight.sh` | Multi-finger loss uncertainty weight | Fig. 5c |
