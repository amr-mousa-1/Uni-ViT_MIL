# UniViT-MIL

## A Unified Vision Transformer and Multi-Instance Learning Framework for Multi-Cancer Histopathology and Biomarker Prediction

Official PyTorch implementation of **UniViT-MIL**, a unified deep learning framework that integrates **Vision Transformers (ViT)** with **Attention-based Multi-Instance Learning (MIL)** for weakly supervised whole-slide image (WSI) analysis.

---

## Abstract

Computational pathology increasingly relies on deep learning for automated cancer diagnosis and biomarker inference from histopathological whole-slide images (WSIs). However, the gigapixel resolution of WSIs and the scarcity of pixel-level annotations pose significant challenges. UniViT-MIL addresses these limitations by combining transformer-based global representation learning with attention-driven multi-instance aggregation. The framework enables accurate multi-cancer classification and biomarker prediction using only slide-level supervision, while maintaining interpretability through attention heatmaps.

---

## Method Overview

The UniViT-MIL pipeline consists of four principal stages:

1. **Patch Extraction and Preprocessing**
   WSIs are segmented to remove background regions, tiled into fixed-size patches, and normalized using standard stain normalization techniques.

2. **Patch Encoding via Vision Transformers**
   Each patch is encoded using a Vision Transformer backbone, enabling long-range dependency modeling across spatially distant tissue regions.

3. **Attention-based Multi-Instance Learning Aggregation**
   Patch embeddings are aggregated using a learnable attention mechanism that assigns higher importance to diagnostically relevant regions.

4. **Slide-level Prediction**
   Aggregated representations are passed to task-specific classification heads for histological subtype classification or biomarker prediction.

---

## Architecture

```
Whole Slide Image (WSI)
        ↓
Patch Extraction and Stain Normalization
        ↓
Vision Transformer Encoder
        ↓
Patch-level Embeddings
        ↓
Attention-based MIL Pooling
        ↓
Slide-level Representation
        ↓
Classification / Biomarker Prediction Head
```

The framework jointly models global tissue context and localized morphological patterns by fusing transformer representations with MIL aggregation.

---

## Repository Structure

```
UniViT_MIL/
│
├── dataset_csv/        # Example datasets and CSV metadata
├── dataset_modules/   # Dataset loaders (generic, HDF5, WSI)
├── models/             # Vision Transformer, MIL, and CLAM-style modules
├── heatmaps/           # Attention-based visualization outputs
├── presets/            # Training and experiment configurations
├── splits/             # Patient-level train/validation/test splits
├── utils/              # Evaluation, transformation, and helper utilities
├── train.py            # Model training script
├── evaluate.py         # Model evaluation script
└── README.md
```

---

## Installation

### Requirements

* Python 3.8 or later
* PyTorch 1.12 or later
* CUDA 11 or later (recommended)

### Setup

```bash
git clone https://github.com/amr-mousa-1/Uni-ViT_MIL.git
cd Uni-ViT_MIL
pip install -r requirements.txt
```

---

## Data Preparation

The framework supports both whole-slide image datasets and pre-extracted feature bags. It has been validated on TCGA, ICIAR2018, and SICAPv2 datasets.

### WSI Preprocessing

```bash
python data/preprocess_wsi.py \
  --source_dir /path/to/raw_slides \
  --output_dir /path/to/processed_patches \
  --patch_size 224 \
  --magnification 20x
```

Preprocessing steps include tissue masking, patch extraction, stain normalization, and class balancing.

---

## Training

### Multi-Cancer Classification

```bash
python train.py \
  --task classification \
  --arch vit_base \
  --mil_type attention \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4
```

### Biomarker Prediction

```bash
python train.py \
  --task biomarker \
  --biomarker_name TMB \
  --cancer_type endometrial \
  --pretrained_weights checkpoints/best_model.pth \
  --epochs 50
```

---

## Evaluation

```bash
python evaluate.py \
  --model_path checkpoints/best_model.pth \
  --test_data /path/to/test_set \
  --save_results results/
```

Reported metrics include Accuracy, Precision, Recall, F1-score, and Area Under the ROC Curve (AUROC).

---

## Experimental Results

| Cancer Type | Task       | Accuracy | AUROC |
| ----------- | ---------- | -------- | ----- |
| Bone        | Subtyping  | 98.4%    | 0.99  |
| Breast      | IDC vs ILC | 96.2%    | 0.97  |
| Endometrial | POLE / TMB | 99.1%    | 0.99  |
| Cervical    | Subtyping  | 97.1%    | 0.98  |

Results are obtained using internal validation on TCGA and publicly available histopathology datasets.

---

## Interpretability

UniViT-MIL provides attention-based heatmaps that highlight diagnostically relevant tissue regions contributing to slide-level predictions. These visualizations support qualitative analysis and clinical interpretability.

---

## Citation

```bibtex
@article{UniViT2025,
  title={UniViT-MIL: A Unified Vision Transformer and Multi-Instance Learning Framework for Multi-Cancer Histopathology},
  author={Abdel-Haii, Amr M. and Mohamed, Malak H.},
  journal={ISEF Projects},
  year={2025}
}
```

---

## Authors

Amr M. Abdel-Haii
[Amr.1424040@stemassiut.moe.edu.eg](mailto:Amr.1424040@stemassiut.moe.edu.eg)

Malak H. Mohamed
[Malak.1424553@stemassiut.moe.edu.eg](mailto:Malak.1424553@stemassiut.moe.edu.eg)

---

## License

This project is licensed under the MIT License.
