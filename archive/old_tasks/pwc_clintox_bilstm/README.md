# ClinTox BiLSTM: Molecular Toxicity Prediction

## Overview

This task focuses on improving the BiLSTM (Bidirectional Long Short-Term Memory) model for predicting clinical toxicity of drug compounds on the ClinTox dataset.

## Task Description

**Objective**: Improve the BiLSTM model to achieve better performance on the ClinTox molecular toxicity prediction task.

**Paper**: "Accelerating Drug Safety Assessment using Bidirectional-LSTM for SMILES Data" (https://arxiv.org/abs/2407.18919v1)

**Dataset**: ClinTox - A molecular toxicity dataset containing SMILES sequences with clinical toxicity labels (FDA-approved vs toxic compounds).

**Model**: BiLSTM - A bidirectional LSTM that processes SMILES sequences to predict drug safety.

## Baseline Performance

The current BiLSTM model achieves the following baseline performance on ClinTox:

- **AUC (Area Under ROC Curve)**: 0.97

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline AUC:

**Scoring**: 
```
Score = (new_auc - 0.97) / 0.97 * 100
```
Example: AUC = 0.98 → Score = (0.98 - 0.97) / 0.97 * 100 = 1.03%

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_clintox_bilstm_main
```

## Training Time

| Training Time |
|---------------|
| 5 minutes |

## Resources Provided

- **Research Paper**: Original BiLSTM paper for drug safety assessment
- **Source Code**: Complete implementation of the BiLSTM model for SMILES data
- **Dataset**: ClinTox dataset with SMILES sequences and toxicity labels
- **Solution Template**: Starting point with BiLSTM architecture and SMILES tokenization

## Technical Requirements

- **GPU**: Recommended for faster training
- **Memory**: 8-16GB RAM
- **Framework**: PyTorch
- **Key Dependencies**: torch, numpy, pandas, scikit-learn, rdkit (for molecular modeling)

## File Structure

```
pwc_clintox_bilstm/
├── pwc_clintox_bilstm.py  # Main task definition
├── manifest.yaml           # Task configuration
├── build_steps.json        # Environment setup
├── requirements.txt        # Dependencies
├── README.md              # This file
└── assets/
    ├── score.py           # Scoring logic
    └── for_agent/
        ├── paper.pdf      # Research paper
        ├── repo.zip       # BiLSTM source code
        ├── dataset.zip    # ClinTox dataset
        └── solution.py    # Implementation template with BiLSTM
```

## Implementation Guidelines

1. **Understand SMILES Representation**:
   - SMILES (Simplified Molecular Input Line Entry System) is a notation for describing molecular structures
   - Example: `CCO` represents ethanol (CH3-CH2-OH)
   - SMILES sequences need tokenization for neural network processing

2. **Study the BiLSTM Architecture**:
   - Bidirectional LSTM processes SMILES sequences forward and backward
   - Captures molecular patterns and substructures
   - Predicts binary toxicity (toxic vs non-toxic)

3. **Analyze the ClinTox Dataset**:
   - FDA-approved drugs vs drugs that failed clinical trials due to toxicity
   - Imbalanced dataset - consider appropriate handling
   - SMILES sequences vary in length

4. **Identify Improvements**:
   - Enhanced SMILES tokenization strategies
   - Better molecular feature representations
   - Improved BiLSTM architecture (layers, hidden dimensions, dropout)
   - Attention mechanisms for focusing on toxic substructures
   - Molecular descriptors and chemical features
   - Data augmentation (SMILES enumeration, molecular scaffolds)
   - Ensemble methods

5. **Molecular Modeling Considerations**:
   - Chemical validity of SMILES strings
   - Molecular fingerprints (Morgan, MACCS, etc.)
   - Drug-target interaction patterns
   - Toxicophore identification
   - Chemical space representation

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, float]:
    # Your implementation here
    # 1. Load and preprocess ClinTox dataset (SMILES + labels)
    # 2. Train improved BiLSTM model
    # 3. Evaluate on test set
    # 4. Calculate AUC
    
    return {
        'auc': your_auc_score  # e.g., 0.98 for 98% AUC
    }
```

## Key Concepts

- **SMILES**: Textual representation of molecular structures
- **BiLSTM**: Bidirectional LSTM for sequential data processing
- **Clinical Toxicity**: Drug compounds that are toxic to humans
- **AUC**: Area Under the Receiver Operating Characteristic Curve - measures classification performance
- **Drug Safety Assessment**: Predicting whether a drug compound is safe for human use

## Example SMILES Sequences

```
CCO                     # Ethanol
CC(C)Cc1ccc(cc1)C(C)C(=O)O  # Ibuprofen
CC(=O)Oc1ccccc1C(=O)O  # Aspirin
```

## Improvement Ideas

1. **Architecture Enhancements**:
   - Multi-layer BiLSTM with residual connections
   - Attention mechanisms over SMILES sequences
   - Hybrid CNN-BiLSTM architecture
   - Graph neural networks for molecular graphs

2. **Feature Engineering**:
   - Molecular fingerprints (Morgan, MACCS)
   - Chemical descriptors (molecular weight, logP)
   - Substructure counts
   - 3D molecular conformations

3. **Training Strategies**:
   - Transfer learning from larger molecular datasets
   - Multi-task learning (toxicity + other properties)
   - Class imbalance handling
   - Cross-validation strategies

## References

- Paper: [Accelerating Drug Safety Assessment using Bidirectional-LSTM for SMILES Data](https://arxiv.org/abs/2407.18919v1)
- ClinTox Dataset: Part of MoleculeNet benchmark
- SMILES Specification: [OpenSMILES](http://opensmiles.org/)
