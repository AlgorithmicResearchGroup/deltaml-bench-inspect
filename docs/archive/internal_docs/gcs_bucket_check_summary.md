# GCS Bucket Check Summary
**Bucket:** `benchmarking_tasks_repos`  
**Date:** October 10, 2025

---

## Step 1: Task Count

**Total Tasks: 54**

All task directories listed in full output.

---

## Step 2: File Structure Analysis

### Standard File Pattern
Most tasks follow a consistent structure with **5 files**:
- `dataset.zip` - The actual dataset (sizes vary widely)
- `metadata.json` - Task metadata (typically 600B-1KB)
- `notebook.ipynb` - Jupyter notebook with implementation
- `paper.pdf` - Research paper
- `repo.zip` - Code repository

### File Presence Across All 54 Tasks

| File Type | Count | Percentage | Status |
|-----------|-------|------------|--------|
| `metadata.json` | 54/54 | 100.0% | ✅ Complete |
| `paper.pdf` | 54/54 | 100.0% | ✅ Complete |
| `repo.zip` | 54/54 | 100.0% | ✅ Complete |
| `dataset.zip` | 52/54 | 96.3% | 🔴 2 missing |
| `notebook.ipynb` | 51/54 | 94.4% | 🟡 3 missing |

**Tasks with Complete File Set (all 5 files): 49/54 (90.7%)**

**Tasks with Missing Files (5 tasks):**
1. **Electricity_(192)_CycleNet** - Missing: `dataset.zip`
2. **MIMIC-III_FLD** - Missing: `dataset.zip`
3. **ETTh1_(336)_Multivariate_AMD** - Missing: `notebook.ipynb`
4. **MNIST_rKAN** - Missing: `notebook.ipynb`
5. **Tiny_ImageNet_Classification_MANO-tiny** - Missing: `notebook.ipynb`

### Complete Dataset Size List (All 54 Tasks - Descending Order)

| Rank | Task Name | Dataset Size |
|------|-----------|--------------|
| 1 | BTAD_URD | 4.90 GB |
| 2 | Food-101_MANO-tiny | 4.65 GB |
| 3 | STL-10,_40_Labels_SemiOccam | 2.44 GB |
| 4 | HME100K_ICAL | 2.11 GB |
| 5 | ISTD+_RASM | 2.00 GB |
| 6 | Stanford_Cars_ProMetaR | 1.83 GB |
| 7 | York_Urban_Dataset_DT-LSD | 1.68 GB |
| 8 | 5-Datasets_CODE-CL | 1.16 GB |
| 9 | CAT2000_SUM | 580.63 MB |
| 10 | Training_and_validation_dataset_of_capsule_vision_2024_challenge._BiomedCLIP+PubmedBERT | 383.57 MB |
| 11 | Weather_(192)_xPatch | 338.64 MB |
| 12 | Tiny-ImageNet_PRO-DSC | 284.94 MB |
| 13 | Tiny_ImageNet_Classification_MANO-tiny | 284.94 MB |
| 14 | ZJU-RGB-P_CSFNet-2 | 281.87 MB |
| 15 | UCR_Anomaly_Archive_KAN | 166.98 MB |
| 16 | CIFAR-10_ABNet-2G-R0 | 162.15 MB |
| 17 | CIFAR-10_ResNet18_(FSGDM) | 162.15 MB |
| 18 | CIFAR-100_PRO-DSC | 160.65 MB |
| 19 | PeMSD4_PM-DMNet(R) | 136.58 MB |
| 20 | ETTh1_(336)_Multivariate_SOFTS | 103.00 MB |
| 21 | Digital_twin-supported_deep_learning_for_fault_diagnosis_DANN | 96.39 MB |
| 22 | Office-31_EUDA | 76.20 MB |
| 23 | MalNet-Tiny_GatedGCN+ | 75.44 MB |
| 24 | FER2013_VGG_based | 74.11 MB |
| 25 | MNIST_rKAN | 63.46 MB |
| 26 | Fashion-MNIST_Continued_fraction_of_straight_lines | 58.90 MB |
| 27 | Fashion-MNIST_ENERGIZE | 58.90 MB |
| 28 | Fashion-MNIST_GECCO | 58.90 MB |
| 29 | ZINC_NeuralWalker | 44.49 MB |
| 30 | ogbl-ddi_GCN_(node_embedding) | 44.29 MB |
| 31 | TXL-PBC:_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n | 42.30 MB |
| 32 | Chameleon_CoED | 31.28 MB |
| 33 | Astock_SRL&Factors | 30.96 MB |
| 34 | WiGesture_CSI-BERT | 25.25 MB |
| 35 | MNIST_GatedGCN+ | 22.02 MB |
| 36 | SumMe_CSTA | 21.35 MB |
| 37 | Kvasir-SEG_EMCAD | 19.07 MB |
| 38 | Kvasir-SEG_EffiSegNet-B5 | 19.07 MB |
| 39 | Kvasir-SEG_Yolo-SAM_2 | 19.07 MB |
| 40 | Gowalla_RLAE-DAN | 14.57 MB |
| 41 | Peptides-struct_GCN+ | 6.07 MB |
| 42 | ETTh1_(720)_Multivariate_SparseTSF | 3.79 MB |
| 43 | FB15k-237_DaBR | 2.11 MB |
| 44 | ogbg-molhiv_GatedGCN+ | 1.97 MB |
| 45 | California_Housing_Prices_Binary_Diffusion | 1.81 MB |
| 46 | ETTh1_(336)_Multivariate_AMD | 405.13 KB |
| 47 | Traffic_GLinear | 395.87 KB |
| 48 | PDBbind_BAPULM | 231.90 KB |
| 49 | MM-Vet_FlashSloth-HD | 180.63 KB |
| 50 | clintox_BiLSTM | 19.40 KB |
| 51 | ImageNet-10_DPAC | 12.11 KB |
| 52 | CNN | 4.84 KB |
| 53 | Electricity_(192)_CycleNet | **Dataset missing** |
| 54 | MIMIC-III_FLD | **Dataset missing** |

---

## Step 3: Task Name vs Metadata Verification

### ✅ CONFIRMED MATCHES (All 54 tasks)

All tasks have metadata.json files present. Here's the complete verification:

| # | Task Folder Name | Metadata Dataset Field | Match Status |
|---|-----------------|----------------------|--------------|
| 1 | `5-Datasets_CODE-CL` | "5-Datasets" | ✅ MATCH |
| 2 | `Astock_SRL&Factors` | "Astock" | ✅ MATCH |
| 3 | `BTAD_URD` | "BTAD" | ✅ MATCH |
| 4 | `CAT2000_SUM` | "CAT2000" | ✅ MATCH |
| 5 | `CIFAR-100_PRO-DSC` | "CIFAR-100" | ✅ MATCH |
| 6 | `CIFAR-10_ABNet-2G-R0` | "CIFAR-10" | ✅ MATCH |
| 7 | `CIFAR-10_ResNet18_(FSGDM)` | "CIFAR-10" | ✅ MATCH |
| 8 | `CNN` | "CNN / Daily Mail" | 🟡 PARTIAL - Missing "/ Daily Mail" |
| 9 | `California_Housing_Prices_Binary_Diffusion` | "California Housing Prices" | ✅ MATCH |
| 10 | `Chameleon_CoED` | "Chameleon" | ✅ MATCH |
| 11 | `Digital_twin-supported_deep_learning_for_fault_diagnosis_DANN` | "Digital twin-supported deep learning for fault diagnosis" | ✅ MATCH |
| 12 | `ETTh1_(336)_Multivariate_AMD` | "ETTh1 (336) Multivariate" | ✅ MATCH |
| 13 | `ETTh1_(336)_Multivariate_SOFTS` | "ETTh1 (336) Multivariate" | ✅ MATCH |
| 14 | `ETTh1_(720)_Multivariate_SparseTSF` | "ETTh1 (720) Multivariate" | ✅ MATCH |
| 15 | `Electricity_(192)_CycleNet` | "Electricity (192)" | ✅ MATCH |
| 16 | `FB15k-237_DaBR` | "FB15k-237" | ✅ MATCH |
| 17 | `FER2013_VGG_based` | "FER2013" | ✅ MATCH |
| 18 | `Fashion-MNIST_Continued_fraction_of_straight_lines` | "Fashion-MNIST" | ✅ MATCH |
| 19 | `Fashion-MNIST_ENERGIZE` | "Fashion-MNIST" | ✅ MATCH |
| 20 | `Fashion-MNIST_GECCO` | "Fashion-MNIST" | ✅ MATCH |
| 21 | `Food-101_MANO-tiny` | "Food-101" | ✅ MATCH |
| 22 | `Gowalla_RLAE-DAN` | "Gowalla" | ✅ MATCH |
| 23 | `HME100K_ICAL` | "HME100K" | ✅ MATCH |
| 24 | `ISTD+_RASM` | "ISTD+" | ✅ MATCH |
| 25 | `ImageNet-10_DPAC` | "ImageNet-10" | ✅ MATCH |
| 26 | `Kvasir-SEG_EMCAD` | "Kvasir-SEG" | ✅ MATCH |
| 27 | `Kvasir-SEG_EffiSegNet-B5` | "Kvasir-SEG" | ✅ MATCH |
| 28 | `Kvasir-SEG_Yolo-SAM_2` | "Kvasir-SEG" | ✅ MATCH |
| 29 | `MIMIC-III_FLD` | "MIMIC-III" | ✅ MATCH |
| 30 | `MM-Vet_FlashSloth-HD` | "MM-Vet" | ✅ MATCH |
| 31 | `MNIST_GatedGCN+` | "MNIST" | ✅ MATCH |
| 32 | `MNIST_rKAN` | "MNIST" | ✅ MATCH |
| 33 | `MalNet-Tiny_GatedGCN+` | "MalNet-Tiny" | ✅ MATCH |
| 34 | `Office-31_EUDA` | "Office-31" | ✅ MATCH |
| 35 | `PDBbind_BAPULM` | "PDBbind" | ✅ MATCH |
| 36 | `PeMSD4_PM-DMNet(R)` | "PeMSD4" | ✅ MATCH |
| 37 | `Peptides-struct_GCN+` | "Peptides-struct" | ✅ MATCH |
| 38 | `STL-10,_40_Labels_SemiOccam` | "STL-10, 40 Labels" | ✅ MATCH |
| 39 | `Stanford_Cars_ProMetaR` | "Stanford Cars" | ✅ MATCH |
| 40 | `SumMe_CSTA` | "SumMe" | ✅ MATCH |
| 41 | `TXL-PBC:_a_freely_accessible_labeled_peripheral_blood_cell_dataset_yolov5n` | "TXL-PBC: a freely accessible labeled peripheral blood cell dataset" | ✅ MATCH |
| 42 | `Tiny-ImageNet_PRO-DSC` | "Tiny-ImageNet" | ✅ MATCH |
| 43 | `Tiny_ImageNet_Classification_MANO-tiny` | "Tiny ImageNet Classification" | ✅ MATCH |
| 44 | `Traffic_GLinear` | "Traffic" | ✅ MATCH |
| 45 | `Training_and_validation_dataset_of_capsule_vision_2024_challenge._BiomedCLIP+PubmedBERT` | "Training and validation dataset of capsule vision 2024 challenge." | ✅ MATCH |
| 46 | `UCR_Anomaly_Archive_KAN` | "UCR Anomaly Archive" | ✅ MATCH |
| 47 | `Weather_(192)_xPatch` | "Weather (192)" | ✅ MATCH |
| 48 | `WiGesture_CSI-BERT` | "WiGesture" | ✅ MATCH |
| 49 | `York_Urban_Dataset_DT-LSD` | "York Urban Dataset" | ✅ MATCH |
| 50 | `ZINC_NeuralWalker` | "ZINC" | ✅ MATCH |
| 51 | `ZJU-RGB-P_CSFNet-2` | "ZJU-RGB-P" | ✅ MATCH |
| 52 | `clintox_BiLSTM` | "clintox" | ✅ MATCH |
| 53 | `ogbg-molhiv_GatedGCN+` | "ogbg-molhiv" | ✅ MATCH |
| 54 | `ogbl-ddi_GCN_(node_embedding)` | "ogbl-ddi" | ✅ MATCH |

---

## Issues Found

### 🔴 Missing Files Summary

**5 tasks are missing required files** (out of 54 total):

#### Missing `dataset.zip` (2 tasks):
1. **Electricity_(192)_CycleNet** - No dataset file present
2. **MIMIC-III_FLD** - No dataset file present

#### Missing `notebook.ipynb` (3 tasks):
1. **ETTh1_(336)_Multivariate_AMD** - No notebook file present
2. **MNIST_rKAN** - No notebook file present
3. **Tiny_ImageNet_Classification_MANO-tiny** - No notebook file present

### 🟡 Minor Naming Discrepancy (1 task)

**Task:** `CNN`
- **Folder name:** `CNN`
- **Metadata dataset:** `CNN / Daily Mail`
- **Issue:** Folder uses abbreviated name "CNN" while metadata specifies full dataset name "CNN / Daily Mail"
- **Severity:** Minor - CNN is present, but missing "/ Daily Mail" suffix

### ✅ All Other Tasks (51/54)

All remaining 51 tasks show proper alignment between folder names and their metadata.json dataset fields with acceptable formatting variations (underscores vs spaces/hyphens).

---

## Data Quality Notes

### File Coverage
✅ **100% coverage (54/54):**
- `metadata.json` - All tasks have metadata
- `paper.pdf` - All tasks have papers
- `repo.zip` - All tasks have code repositories

🔴 **Incomplete coverage:**
- `dataset.zip` - 52/54 tasks (96.3%); **2 tasks missing**
- `notebook.ipynb` - 51/54 tasks (94.4%); **3 tasks missing**

✅ **Complete file sets** - 49/54 tasks (90.7%) have all 5 required files  
✅ **Valid JSON** - All metadata.json files parsed successfully  
✅ **Complete metadata** - All metadata files contain dataset, model_name, paper_title, and metrics fields  
🟡 **Naming alignment** - 53/54 tasks have perfect matches; 1 task (CNN) has minor naming discrepancy  

### Summary Statistics
- **Total tasks:** 54
- **Tasks with complete file sets:** 49 (90.7%)
- **Tasks missing files:** 5 (9.3%)
  - Missing dataset.zip: 2 tasks
  - Missing notebook.ipynb: 3 tasks
- **Largest dataset:** BTAD_URD (4.90 GB)
- **Smallest dataset:** CNN (4.84 KB)
- **Total dataset storage:** ~20+ GB (across all tasks with datasets)  

---

## Full Output
Complete detailed output saved to: `gcs_bucket_check_results.txt`

