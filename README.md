# FER-2013 Dataset Setup Guide

This folder holds the dataset structure for training and evaluating the MoodTune AI CNN model.

## Dataset Structure

Place the FER-2013 dataset in the following directory layout:

```text
dataset/
└── fer2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
```

## How to Obtain FER-2013

1. Download the dataset from Kaggle:
   [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. Extract the archive into `dataset/fer2013/`.
3. Verify that all 7 emotion subfolders exist under both `train` and `test`.

## Dataset Specifications

- **Total Images**: ~35,887 images
- **Image Format**: Grayscale 48 × 48 PNG/JPG images
- **Classes**: 7 emotions (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`)
- **Usage**: Used by `train_model.py` and `evaluate_model.py`.
