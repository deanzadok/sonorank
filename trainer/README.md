# Trainer

Core training code for SonoRank and baseline models.

## Training pipeline

The SonoRank pipeline trains in three levels, each building on the previous:

| Level | Script | Model | Purpose |
|-------|--------|-------|---------|
| 1 | `train_embedding.py` | `EmbeddingAutoencoder` | Unsupervised encoder pretraining via image reconstruction |
| 2 | `train_multi_finger_ranking.py` | `MultiFingerTemporalRankingModel` | Temporal ranking with 5 parallel comparison heads |
| 3 | `train_flex_embedding.py` | `FlexEmbeddingClassifier` | Flex classification initialized from the ranking backbone |

### Baselines

| Script | Model | Reference |
|--------|-------|-----------|
| `train_cbmf.py` | `CBMFModel` (UNet+GRU) | Zadok et al. 2025 |
| `train_bimbraw.py` | `BimbrawVGG` (Modified VGG16) | Bimbraw et al. 2023 |
| `train_classic_baselines.py` | sklearn classifiers | Castellini 2011, Huang 2016, McIntosh 2017 |

## Module files

- **model.py:** model classes and shared building blocks
- **load.py:** `SegmentationDataset`, subject splits, HDF5 loading, sampling, and transforms
- **utils.py:** `SummaryLogger` for concise training logs
- **metrics.py:** retained segmentation metrics used by legacy components

## Usage

All trainers are invoked as modules from the project root:

```bash
python -m trainer.train_multi_finger_ranking --gpu-id 0 --help
```

Configuration is passed through command-line arguments. Each run saves its settings as `config.json` in the output directory.

## Two-phase training

Levels 2 and 3 support two phases:

- **Phase 1:** freeze the encoder and train the newly added layers.
- **Phase 2:** fine-tune the selected parts of the full model with separate learning rates.

The phase lengths and learning rates are launcher settings. Check the selected shell script and its saved `config.json` rather than assuming one schedule for every experiment.
