# Uni-ViT-MIL Dataset CSV Format Documentation

## Overview

This document describes the expected CSV format for Uni-ViT-MIL datasets.
The example files `univit_example.csv` and `univit_mil_example.csv` demonstrate
this format with synthetic/placeholder data.

**IMPORTANT**: These example files contain FAKE DATA for format demonstration only.
They are NOT real datasets and should be replaced with actual patient data.

---

## Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `case_id` | string | Patient/case identifier. Used for patient-level stratification during train/val/test splitting. Multiple slides can share the same case_id. | `patient_001` |
| `slide_id` | string | Unique identifier for each slide. Must match the filename (without extension) of the corresponding `.pt` feature file. | `slide_001` |
| `label` | string | Cancer classification label. Will be converted to integer via `label_dict` in main.py. | `tumor_tissue`, `normal_tissue` |
| `biomarker_label` | string | Biomarker status label. Will be converted to integer via `biomarker_dict` (auto-generated if not provided). | `TP53_mutant`, `KRAS_wildtype` |

---

## Label Encoding

### Classification Labels (`label` column)

Labels are encoded via dictionaries in `main.py`. Example:

```python
label_dict = {'normal_tissue': 0, 'tumor_tissue': 1}
```

The string values in the CSV are mapped to integers during dataset loading.

### Biomarker Labels (`biomarker_label` column)

Biomarker labels can be:

1. **Auto-encoded**: If `biomarker_dict={}` (empty dict) is passed, unique values
   are automatically extracted and assigned integer indices.

2. **Explicitly encoded**: Provide a dictionary mapping, e.g.:
   ```python
   biomarker_dict = {
       'TP53_wildtype': 0,
       'TP53_mutant': 1,
       'KRAS_wildtype': 2,
       'KRAS_mutant': 3,
       'EGFR_wildtype': 4,
       'EGFR_mutant': 5
   }
   ```

---

## File Structure Requirements

For each `slide_id` in the CSV, a corresponding feature file must exist:

```
data_root_dir/
└── {task}_features/
    └── pt_files/
        ├── slide_001.pt    # Matches slide_id='slide_001'
        ├── slide_002.pt    # Matches slide_id='slide_002'
        └── ...
```

Each `.pt` file should contain a PyTorch tensor of shape `(num_patches, embed_dim)`.

---

## Example CSV Content

```csv
case_id,slide_id,label,biomarker_label
patient_001,slide_001,tumor_tissue,TP53_mutant
patient_001,slide_002,tumor_tissue,TP53_mutant
patient_002,slide_003,normal_tissue,TP53_wildtype
patient_003,slide_004,tumor_tissue,KRAS_mutant
patient_004,slide_005,normal_tissue,EGFR_wildtype
```

### Notes on the Example:

- `patient_001` has 2 slides (slide_001, slide_002) - both tumor, both TP53 mutant
- Patient-level splitting ensures all slides from one patient stay in the same split
- Different patients can have different biomarker statuses

---

## Missing Biomarker Labels

If the `biomarker_label` column is missing from your CSV:

- The dataset will print a warning
- All biomarker labels will default to `0`
- Classification training will still work, but biomarker predictions will be meaningless

To add biomarker labels to an existing CSV, simply add the `biomarker_label` column.

---

## Creating Your Own Dataset

1. **Prepare your CSV** with all four columns
2. **Extract features** using CLAM's `extract_features_fp.py` or similar
3. **Ensure slide_id matching** between CSV and .pt filenames
4. **Define label dictionaries** in `main.py` or let them auto-generate
5. **Run training** with `--model_type univit_mil`

---

*This documentation accompanies the Uni-ViT-MIL research skeleton.*
*Last updated: December 2024*

