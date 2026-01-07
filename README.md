# UniViT-MIL: A Unified Vision Transformer and Multi-Instance Learning Framework

**Official PyTorch Implementation**

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Status](https://img.shields.io/badge/Status-Research_Prototype-green)

## Overview

Modern computational pathology workflows primarily rely on Convolutional Neural Networks (CNNs), which often struggle to capture global morphological context due to limited receptive fields. Furthermore, the prediction of genomic biomarkers (e.g., TMB, MSI) typically requires expensive next-generation sequencing (NGS), limiting accessibility in resource-constrained settings.

**UniViT-MIL** presents a unified deep learning framework that integrates a **Vision Transformer (ViT)** backbone with **Attention-based Multi-Instance Learning (MIL)**. This architecture is designed to perform weakly-supervised multi-cancer classification and genomic biomarker prediction directly from standard H&E-stained Whole Slide Images (WSIs).



By leveraging the self-attention mechanisms of Transformers, UniViT-MIL captures long-range dependencies across tissue slides, enabling:
1.  **Multi-Organ Classification:** Accurate subtyping for Bone, Breast, Cervical, Prostate, and Endometrial cancers.
2.  **Virtual Biopsy:** Prediction of high-cost biomarkers (e.g., *POLE* mutations, High TMB) from morphological features alone.
3.  **Interpretability:** Generation of high-resolution attention heatmaps to localize diagnostically relevant regions.

## Key Contributions

* **Unified Architecture:** A single framework capable of generalizing across five distinct cancer types without architecture modification.
* **Global Context Awareness:** Utilizes Vision Transformers to overcome the locality bias of CNNs, resulting in superior feature extraction from gigapixel-resolution images.
* **Weakly Supervised Learning:** Operates efficiently using only slide-level labels, eliminating the need for expensive pixel-level annotations.
* **Morphological-to-Molecular Mapping:** Demonstrates a significant correlation between tissue morphology and genomic status, offering a cost-effective alternative to sequencing.

## Methodology

The framework operates in a multi-stage pipeline:

1.  **Preprocessing:** WSIs are tiled into non-overlapping patches. Background subtraction and Macenko stain normalization are applied to ensure consistency.
2.  **Feature Encoding:** A pre-trained Vision Transformer processes patches to generate high-dimensional embeddings, capturing global semantic features.
3.  **MIL Aggregation:** An attention-based pooling mechanism aggregates patch embeddings into a single slide-level representation, assigning higher weights to tumor-rich regions.
4.  **Prediction:** Dual classification heads output the histological subtype and, where applicable, the genomic biomarker status.

## Getting Started

### Prerequisites

* Linux or macOS
* Python 3.8+
* PyTorch 1.12+
* CUDA 11.0+ (recommended for GPU acceleration)

### Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YourUsername/UniViT-MIL.git](https://github.com/YourUsername/UniViT-MIL.git)
cd UniViT-MIL
pip install -r requirements.txt
Data Preparation
The model expects Whole Slide Images (WSIs) or pre-extracted feature bags. The code is compatible with standard datasets such as TCGA, PANDA, and ICIAR.

To process raw WSIs into patches:

Bash

python data/preprocess_wsi.py \
    --source_dir /path/to/raw_slides \
    --output_dir /path/to/processed_patches \
    --patch_size 256 \
    --magnification 20x
Training
The training process utilizes a combined loss function (Cross-Entropy + Contrastive Loss) to optimize the model.

1. Multi-Cancer Classification
To train the model on the multi-organ dataset:

Bash

python train.py \
    --task classification \
    --data_root /path/to/data \
    --arch vit_base \
    --mil_type attention \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --save_dir ./checkpoints/classification

2. Biomarker Prediction (e.g., Endometrial TMB)
To fine-tune the model for biomarker prediction tasks:

Bash

python train.py \
    --task biomarker \
    --biomarker_name TMB \
    --cancer_type endometrial \
    --pretrained_weights ./checkpoints/classification/best_model.pth \
    --epochs 50 \
    --lr 5e-5

Evaluation
To evaluate the model on a test set and generate performance metrics (AUC, F1-Score, Accuracy):

Bash

python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --test_data /path/to/test_set \
    --save_results ./results/

Performance BenchmarksCancer TypeTaskAccuracyAUCBoneSubtyping (Osteosarcoma)98.4%0.99EndometrialBiomarker (POLE/TMB)99.1%0.99BreastSubtyping (IDC/ILC)96.2%0.97CervicalSubtyping97.1%0.98Note: Results based on internal validation using TCGA and private datasets.

Citation
If you find this code or research useful, please cite our work:

Code snippet

@article{UniViT2025,
  title={UniViT-MIL: A Unified Vision Transformer and Multi-Instance Learning Framework for Multi-Cancer Histopathology},
  author={Abdel-Haii, Amr M. and Mohamed, Malak H.},
  journal={ISEF Projects},
  year={2025}
}

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For questions or inquiries regarding the implementation, please contact:

Amr M. Abdel-Haii - Amr.1424040@stemassiut.moe.edu.eg

Malak H. Mohamed - Malak.1424553@stemassiut.moe.edu.eg
