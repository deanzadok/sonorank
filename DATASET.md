# Dataset

SonoRank is trained and evaluated on synchronized forearm ultrasound video and finger kinematics recorded from 12 healthy subjects. The isolated finger-flexion subset used by this repository is available separately from the code.

## Download

Download the SonoRank data from the [Technion SharePoint folder](https://technionmail-my.sharepoint.com/:f:/g/personal/deanzadok_campus_technion_ac_il/IgD3zPBSBLfcQ7vAaGkGpZt4Aa8Oh-Ym77CR3S6UZdBBGGs?e=VknQDY).

Direct download-folder URL:

```text
https://technionmail-my.sharepoint.com/:f:/g/personal/deanzadok_campus_technion_ac_il/IgD3zPBSBLfcQ7vAaGkGpZt4Aa8Oh-Ym77CR3S6UZdBBGGs?e=VknQDY
```

The download contains 51 HDF5 recordings and 51 corresponding hand-corrected label files. The complete download is approximately 12.9 GB.

For questions about the dataset, contact Dean Zadok at [deanzadok@cs.technion.ac.il](mailto:deanzadok@cs.technion.ac.il).

## What the dataset contains

- **12 healthy subjects** (IDs 1, 4, 5, 6, 10, 17, 19, 20, 21, 23, 24, 25).
- **B-mode forearm ultrasound** video captured with a wireless Clarius probe.
- **Per-finger joint kinematics** recorded with motion capture (joint angles, velocities, accelerations, and 3D marker positions).
- **51 isolated finger-flexion recordings** containing 83,942 ultrasound frames.
- Hand-corrected per-finger flex-event labels for the finger-flexion sessions.

Each recording is distributed as an HDF5 file containing the ultrasound frames, synchronized timestamps, and kinematic measurements. Hand-corrected flexion labels are distributed as separate JSON files with matching filename stems. Label intervals follow the half-open convention `[start_frame, end_frame)`.

## What SonoRank uses

SonoRank trains only on the **isolated finger-flexion** sessions with hand-corrected flex-event labels.

The download is already organized using the directory structure expected by the code:

```text
data/
└── processed_h5/
    └── h5_files/
        └── recordings_*_gestures_enrollment*_finger_flex*.h5

labeling/
└── fixed_labels/
    └── recordings_*_gestures_enrollment*_finger_flex*.json
```

- `data/processed_h5/h5_files/` contains the 51 recording files. Each HDF5 file stores the ultrasound frames and synchronized kinematic measurements for one finger-flexion session.
- `labeling/fixed_labels/` contains the 51 hand-corrected label files. Each JSON filename has the same stem as its corresponding HDF5 filename.

Place the downloaded top-level `data/` and `labeling/` directories directly inside the SonoRank repository root. Do not place the entire download inside another nested directory. For example, the final path to an HDF5 file should be `sonorank/data/processed_h5/h5_files/recordings_...h5`, and its labels should be at `sonorank/labeling/fixed_labels/recordings_...json`.

The training and evaluation scripts discover the HDF5 recordings and fixed labels from these locations automatically. The naming convention is documented in [`data_utils/README.md`](data_utils/README.md), and the standard 12-fold leave-one-subject-out protocol is described in [`CROSS_SUBJECT_VALIDATION.md`](CROSS_SUBJECT_VALIDATION.md).
