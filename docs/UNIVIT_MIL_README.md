# Uni-ViT-MIL: Unified Vision Transformer for Multiple Instance Learning

## Research Skeleton Documentation

This document provides a comprehensive overview of the Uni-ViT-MIL research model architecture, 
explaining its design philosophy, differences from CLAM, and how to use it for research purposes.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Differences from CLAM](#key-differences-from-clam)
3. [Architecture](#architecture)
4. [File Structure](#file-structure)
5. [Expected Data Format](#expected-data-format)
6. [Model Configuration](#model-configuration)
7. [Future Integration Points](#future-integration-points)

---

## Overview

**Uni-ViT-MIL** (Unified Vision Transformer for Multiple Instance Learning) is a multi-task 
learning model designed for whole slide image (WSI) analysis. Unlike CLAM, which uses 
pre-extracted features from frozen encoders, Uni-ViT-MIL is designed to integrate the 
Vision Transformer as a **trainable component** of the MIL pipeline.

### Research Goals

1. **End-to-end learning**: The ViT encoder can be fine-tuned jointly with the MIL aggregation
2. **Multi-task prediction**: Simultaneous cancer classification and biomarker prediction
3. **Interpretable attention**: Multi-head attention pooling provides richer attention patterns

---

## Key Differences from CLAM

| Aspect | CLAM | Uni-ViT-MIL |
|--------|------|-------------|
| **Feature Extractor** | Frozen (ResNet/ViT) | Trainable ViT (optional) |
| **MIL Pooling** | Gated attention (single/multi-branch) | Multi-head attention |
| **Loss Functions** | Bag loss + Instance clustering loss | Classification + Biomarker loss |
| **Output** | Single classification | Multi-task (classification + biomarker) |
| **Attention Heads** | 1 (SB) or n_classes (MB) | Configurable (default: 8) |

### What Uni-ViT-MIL Does NOT Use

- ❌ Instance-level clustering losses (CLAM's `inst_loss`)
- ❌ SVM-based losses
- ❌ Positive/negative patch sampling (CLAM's `k_sample`)
- ❌ Subtyping-specific instance evaluation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Uni-ViT-MIL                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Input: Bag of Patch Embeddings                          │   │
│  │ Shape: (num_patches, embed_dim)                         │   │
│  │ Source: Pre-extracted (UNI, ResNet) or raw patches      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Vision Transformer Encoder (Optional)                   │   │
│  │ - Only used if processing raw patch images              │   │
│  │ - Can be initialized from pretrained weights (e.g., UNI)│   │
│  │ - Trainable or frozen based on configuration            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Multi-Head Attention MIL Pooling                        │   │
│  │ - Learnable query vectors aggregate patch information   │   │
│  │ - Multiple attention heads capture diverse patterns     │   │
│  │ - Output: Single slide-level embedding                  │   │
│  │                                                         │   │
│  │ Attention: softmax(Q·K^T / √d) · V                     │   │
│  │ Where Q = learnable queries, K,V = patch embeddings    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                  │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │ Classification Head │         │ Biomarker Head      │       │
│  │ - Linear layers     │         │ - Linear layers     │       │
│  │ - n_classes outputs │         │ - n_biomarkers out  │       │
│  └─────────────────────┘         └─────────────────────┘       │
│              │                               │                  │
│              ▼                               ▼                  │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │ classification_logits│        │ biomarker_logits    │       │
│  └─────────────────────┘         └─────────────────────┘       │
│                                                                 │
│  Output: {                                                      │
│    'classification_logits': (batch, n_classes),                │
│    'biomarker_logits': (batch, n_biomarkers),                  │
│    'attention_weights': (batch, num_heads, num_queries, N)     │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
Uni-ViT_MIL/
├── models/
│   ├── __init__.py                 # Exports Uni-ViT-MIL classes
│   ├── univit_mil.py              # ⭐ Main model implementation
│   ├── model_clam.py              # Original CLAM (unchanged)
│   └── model_mil.py               # Original MIL (unchanged)
│
├── dataset_modules/
│   ├── dataset_generic.py         # Extended with Generic_MIL_Dataset_MultiTask
│   └── ...
│
├── dataset_csv/
│   ├── univit_example.csv         # ⭐ Example data format
│   └── ...
│
├── utils/
│   ├── core_utils.py              # Extended with train_loop_univit, validate_univit
│   └── eval_utils.py              # Extended with eval_univit, summary_univit
│
├── vis_utils/
│   └── heatmap_utils.py           # Extended with univit attention heatmaps
│
├── heatmaps/
│   └── configs/
│       └── config_univit_template.yaml  # ⭐ Heatmap config for Uni-ViT-MIL
│
├── main.py                        # Extended with univit_mil model_type
├── eval.py                        # Extended with univit_mil evaluation
└── create_heatmaps.py             # Extended with univit_mil heatmaps
```

---

## Expected Data Format

### CSV Structure

The dataset CSV must include these columns for Uni-ViT-MIL:

| Column | Description | Example |
|--------|-------------|---------|
| `case_id` | Patient/case identifier | `patient_001` |
| `slide_id` | Unique slide identifier | `slide_001` |
| `label` | Cancer classification label (string) | `tumor_tissue` |
| `biomarker_label` | Biomarker status (string) | `TP53_mutant` |

### Example CSV (`dataset_csv/univit_example.csv`)

```csv
case_id,slide_id,label,biomarker_label
patient_001,slide_001,tumor_tissue,TP53_mutant
patient_001,slide_002,tumor_tissue,TP53_mutant
patient_002,slide_003,normal_tissue,TP53_wildtype
patient_003,slide_004,tumor_tissue,KRAS_mutant
patient_004,slide_005,normal_tissue,EGFR_wildtype
```

### Feature Files

Pre-extracted features should be stored as:
```
data_root_dir/
└── {task}_features/
    └── pt_files/
        ├── slide_001.pt
        ├── slide_002.pt
        └── ...
```

Each `.pt` file contains a tensor of shape `(num_patches, embed_dim)`.

---

## Model Configuration

### Key Arguments

```python
# Model architecture
--model_type univit_mil          # Use Uni-ViT-MIL
--embed_dim 1024                 # Embedding dimension (match feature extractor)
--n_classes 2                    # Number of classification classes
--n_biomarkers 4                 # Number of biomarker classes
--mil_num_heads 8                # Number of attention heads in MIL pooling

# Loss weighting
--alpha 0.5                      # Weight for biomarker loss
                                 # Total = cls_loss + alpha * bio_loss

# Data
--biomarker_col biomarker_label  # Column name in CSV for biomarker labels
--biomarker_multi_label          # Flag for multi-label biomarker prediction
```

### Model Variants

| Factory Function | Configuration | Parameters |
|------------------|---------------|------------|
| `univit_mil_base()` | ViT-Base | 768-dim, 12 layers, 12 heads |
| `univit_mil_small()` | ViT-Small | 384-dim, 12 layers, 6 heads |
| `univit_mil_large()` | ViT-Large (UNI-style) | 1024-dim, 24 layers, 16 heads |
| `univit_mil_from_embeddings()` | Embeddings only | No ViT encoder, just MIL pooling |

---

## Future Integration Points

### Where to Connect Real Data

1. **Dataset CSV**: Replace `univit_example.csv` with your real data
2. **Feature Directory**: Point `--data_root_dir` to your extracted features
3. **Label Dictionaries**: Update `label_dict` and `biomarker_dict` in `main.py`

### Where to Add Custom Logic

1. **Custom Feature Extractors**: Modify `models/builder.py`
2. **Custom Loss Functions**: Add to `utils/core_utils.py` in `train_loop_univit()`
3. **Custom Metrics**: Extend `utils/eval_utils.py` in `summary_univit()`

### Checkpoints and Pretrained Weights

To load pretrained ViT weights:

```python
from models.univit_mil import UniViTMIL

model = UniViTMIL.from_pretrained_vit(
    vit_checkpoint_path='path/to/vit_weights.pth',
    n_classes=2,
    n_biomarkers=4
)
```

---

## Usage Examples

### Instantiate Model Only (No Training)

```python
from models.univit_mil import univit_mil_from_embeddings

# Create model for pre-extracted embeddings
model = univit_mil_from_embeddings(
    embed_dim=1024,        # Must match your feature extractor
    n_classes=2,           # Cancer classification classes
    n_biomarkers=4,        # Biomarker prediction classes
    mil_num_heads=8,       # Attention heads for MIL pooling
    head_dropout=0.25
)

# Forward pass (for testing)
import torch
dummy_features = torch.randn(100, 1024)  # 100 patches, 1024-dim
outputs = model(dummy_features)

print(f"Classification logits: {outputs['classification_logits'].shape}")
print(f"Biomarker logits: {outputs['biomarker_logits'].shape}")
print(f"Attention weights: {outputs['attention_weights'].shape}")
```

### Command Line (When Ready for Training)

```bash
python main.py \
    --model_type univit_mil \
    --task task_1_tumor_vs_normal \
    --data_root_dir /path/to/features \
    --n_classes 2 \
    --n_biomarkers 4 \
    --alpha 0.5 \
    --exp_code univit_experiment
```

---

## Summary

Uni-ViT-MIL provides a clean research skeleton for exploring:

1. **Multi-head attention MIL** - Richer attention patterns than CLAM
2. **Multi-task learning** - Joint classification and biomarker prediction
3. **Flexible architecture** - Works with pre-extracted features or raw patches
4. **Interpretability** - Per-head attention visualization

This skeleton is designed for **code review and research discussion**. 
The architecture is fully defined but does not require immediate training.

---

*Last updated: December 2024*

