# Data Utilities

Scripts for preparing the dataset from raw recordings to training-ready H5 files.

## Pipeline

```
Raw ultrasound frames + segmentation masks
  → prepare_data.py (resize to 224x224, validate masks)
  → process_h5.py / process_all_h5.py (extract joint angles from H5 kinematics)
  → merge_datasets.py (combine images, masks, and kinematics into unified H5 files)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `prepare_data.py` | Resize images and masks to target resolution, save as pickle files |
| `process_h5.py` | Extract and process joint angle data from raw H5 kinematics files |
| `process_all_h5.py` | Batch-process all H5 files for all subjects |
| `process_all_gestures_h5.py` | Process all gesture types (finger_flex, playing, etc.) |
| `merge_datasets.py` | Merge processed kinematics with image/mask pickle files into final H5 |
| `align_pkl_with_raw.py` | Align pickle frame indices with raw recording timestamps |
| `copy_segmentations_to_raw.py` | Copy segmentation masks into raw data directories |
| `find_alignment.py` | Search for correct frame alignment between data sources |
| `inspect_data.py` | Inspect and print statistics about processed data files |
| `verify_data_split.py` | Verify that subject splits have no data leakage |

## Output format

Final H5 files are placed in `data/processed_h5/h5_files/` with naming:

```
recordings_{subject:02d}_gestures_enrollment{ee}_finger_flex{ff}.h5
```

Each file contains RGB images (224x224), 8-class segmentation masks, and joint angles for 5 fingers.
