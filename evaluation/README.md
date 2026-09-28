# Evaluation

Scripts for evaluating trained models and generating publication figures.

## Evaluation pipeline

The evaluation pipeline produces event-level flexion metrics in three steps:

1. **Generate predictions:** run a trained model on each test session and save the finger-flexion probabilities.
2. **Detect events:** threshold the probability signals and apply the selected duration and gap settings.
3. **Compute metrics:** match the predicted events with the ground-truth intervals.

### Per-method evaluation

```bash
# SonoRank (FlexEmbeddingClassifier)
python -m evaluation.evaluate_flex_embedding_events --checkpoint path/to/checkpoint --gpu-id 0

# Ranking backbone (MultiFingerTemporalRankingModel)
python -m evaluation.evaluate_checkpoint_events --checkpoint path/to/checkpoint --gpu-id 0

# CBMF baseline
python -m evaluation.evaluate_cbmf_events --checkpoint path/to/checkpoint --gpu-id 0
```

### Fair comparison across methods

```bash
python -m evaluation.evaluate_methods_fair --methods-dir outputs/experiments/methods_fixed_labels
```

The script loads the saved fold predictions and evaluates each method using the threshold strategy defined in `evaluate_methods_fair.py`. The output records the settings used for the comparison.

### Recompute metrics from saved predictions

```bash
python -m evaluation.recompute_event_metrics --sweep-dir outputs/experiments/seqlen_sweep/stride2
```

## Core scripts

| Script | Purpose |
|--------|---------|
| `classify_with_ranking_references.py` | Generate P(query > rest) signals using ranking model |
| `event_detection_metrics.py` | Event-level matching and F1/precision/recall computation |
| `evaluate_checkpoint_events.py` | End-to-end evaluation for ranking checkpoints |
| `evaluate_flex_embedding_events.py` | End-to-end evaluation for FlexEmbeddingClassifier |
| `evaluate_cbmf_events.py` | End-to-end evaluation for CBMF baseline |
| `evaluate_methods_fair.py` | Fair comparison across all methods with shared thresholds |
| `recompute_event_metrics.py` | Recompute event metrics from saved raw predictions |
| `benchmark_realtime.py` | Measure real-time inference latency and throughput |

## Visualization

| Script | Purpose |
|--------|---------|
| `visualize_flex_embedding.py` | Plot per-finger P(flex) predictions vs ground truth |
| `visualize_flex_embedding_events.py` | Plot detected events with TP/FP/FN shading |

## Publication figures (`plots/`)

| Script | Paper figure |
|--------|-------------|
| `plot_methods_scatter.py` | Grouped bar chart comparing all methods |
| `plot_method_comparison.py` | Single-session method comparison panels |
| `plot_method_rollout.py` | Temporal rollout of predictions vs ground truth |
| `plot_qualitative_panels.py` | Four-panel qualitative results figure |
| `plot_per_subject_auc.py` | Per-subject per-finger AUC distribution |
| `plot_seqlen_sweep.py` | Sequence length ablation results |
| `plot_subject_count_sweep.py` | Training set size ablation results |
