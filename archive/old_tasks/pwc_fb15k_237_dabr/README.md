# DaBR: Distance-Adaptive Quaternion Knowledge Graph Embedding on FB15k-237

## Overview

This task focuses on improving the DaBR model for knowledge graph completion on the FB15k-237 dataset using quaternion-based embeddings with bidirectional rotation.

## Task Description

**Objective**: Improve the DaBR model to achieve better link prediction performance on FB15k-237 using distance-adaptive quaternion embeddings.

**Paper**: "Distance-Adaptive Quaternion Knowledge Graph Embedding with Bidirectional Rotation" (https://arxiv.org/abs/2412.04076v2)

**Dataset**: FB15k-237 - A knowledge graph benchmark derived from Freebase with inverse relations removed.

**Model**: DaBR - A quaternion-based knowledge graph embedding model that uses distance-adaptive mechanisms and bidirectional rotation operations.

## Baseline Performance

The current DaBR model achieves the following baseline performance on FB15k-237:

- **Hits@1**: 0.247 (24.7%)
- **Hits@3**: 0.410 (41.0%)
- **Hits@10**: 0.572 (57.2%)
- **MRR (Mean Reciprocal Rank)**: 0.373
- **MR (Mean Rank)**: 83

## Evaluation Metrics

The task is evaluated based on percentage improvement over the baseline Hits@1 (**higher is better**):

**Scoring**: 
```
Score = (new_hits@1 - baseline_hits@1) / baseline_hits@1 * 100
```
Example: Hits@1 = 0.270 → Score = (0.270 - 0.247) / 0.247 * 100 = 9.31%

**Note**: For Hits@1, Hits@3, Hits@10, and MRR, **higher values are better**. For MR (Mean Rank), **lower values are better**. Other metrics are reported for analysis but do not affect the score.

## Training Time

**Estimated Training Time**: 114 hours (A100 GPU), 38 hours (H100 GPU)
- **Configuration**: 114 hours for 10K epochs for A100 GPU, 38 hours if using H100 GPU
- **Previous verification**: No training time provided in paper or repo. All experiments were conducted on a single NVIDIA GeForce RTX 4090 GPU with 24GB of memory according to paper

## Resources Provided

- **Research Paper**: Original DaBR paper on distance-adaptive quaternion embeddings
- **Source Code**: Complete implementation of the DaBR model (repository not publicly available)
- **Dataset**: FB15k-237 will be downloaded automatically during setup
- **Solution Template**: Starting point with model architecture and evaluation logic

## Technical Requirements

- **GPU**: Recommended for faster training
- **Memory**: 8-16GB RAM
- **Framework**: PyTorch
- **Key Dependencies**: torch, numpy, pandas, scipy

## File Structure

```
pwc_fb15k_237_dabr/
├── pwc_fb15k_237_dabr.py  # Main task definition
├── manifest.yaml          # Task configuration
├── build_steps.json       # Environment setup
├── requirements.txt       # Dependencies
├── README.md              # This file
└── assets/
    ├── score.py           # Scoring logic
    └── for_agent/
        ├── paper.pdf      # Research paper
        ├── repo.zip       # DaBR source code
        └── solution.py    # Implementation template
```

## Implementation Guidelines

1. **Understand the FB15k-237 Dataset**:
   - Knowledge graph benchmark from Freebase
   - 14,541 entities
   - 237 relation types
   - 272,115 training triples (head, relation, tail)
   - 17,535 validation triples
   - 20,466 test triples
   - Inverse relations removed (harder than FB15k)
   - Task: Link prediction - predict missing entities

2. **Study the DaBR Model**:
   - **Quaternion Embeddings**: 4D hypercomplex numbers (w, x, y, z)
   - **Distance-Adaptive**: Adjust embedding behavior based on entity distance
   - **Bidirectional Rotation**: Model relations as rotations in both directions
   - **Hamilton Product**: Quaternion multiplication for relation composition
   - **Scoring Function**: Measure plausibility of triples using quaternion operations

3. **Analyze Knowledge Graph Embedding**:
   - Entities and relations embedded in continuous vector space
   - Triples (h, r, t) should have high scores if valid
   - Learn to model various relation patterns:
     - 1-to-1: e.g., "has_capital"
     - 1-to-N: e.g., "has_city"
     - N-to-1: e.g., "capital_of"
     - N-to-N: e.g., "works_in"
   - Composition and symmetry of relations

4. **Identify Improvements**:
   - Enhanced quaternion rotation mechanisms
   - Better distance-adaptive strategies
   - Improved negative sampling techniques
   - Advanced regularization for quaternions
   - Relation-aware embeddings
   - Multi-hop reasoning
   - Data augmentation (inverse relations, composition)
   - Ensemble methods

5. **Knowledge Graph Specific Considerations**:
   - **Training**: Use negative sampling to generate false triples
   - **Evaluation**: Rank candidate entities for test triples
   - **Metrics**:
     - Hits@k: Percentage of correct entities in top-k predictions
     - MRR: Mean of reciprocal ranks of correct entities
     - MR: Mean rank of correct entities
   - **Filtered Setting**: Remove other valid triples during ranking
   - **Batch Processing**: Handle large graphs efficiently

## Validation

The solution will be automatically evaluated using the provided scoring script. Ensure your implementation follows the expected interface:

```python
def evaluate() -> Dict[str, Union[float, int]]:
    # Your implementation here
    # 1. Load FB15k-237 dataset
    # 2. Train improved DaBR model
    # 3. Evaluate on test set with link prediction
    # 4. Calculate metrics
    
    return {
        'hits@1': your_hits_at_1,    # e.g., 0.270 (higher is better)
        'hits@3': your_hits_at_3,    # e.g., 0.430 (higher is better)
        'hits@10': your_hits_at_10,  # e.g., 0.590 (higher is better)
        'mrr': your_mrr,             # e.g., 0.390 (higher is better)
        'mr': your_mr                # e.g., 75 (lower is better)
    }
```

## Key Concepts

- **Knowledge Graph**: Graph structure representing entities and relations as (head, relation, tail) triples
- **Link Prediction**: Predicting missing entities in incomplete triples
- **Quaternion**: 4D hypercomplex number with rich algebraic properties
- **Hamilton Product**: Quaternion multiplication rule for composing rotations
- **Distance-Adaptive**: Adjusting model behavior based on distance metrics
- **Bidirectional Rotation**: Modeling relations as rotations in both forward and backward directions
- **Hits@k**: Percentage of correct predictions in top-k ranked candidates
- **MRR**: Average of 1/rank for correct entities
- **Negative Sampling**: Generating false triples for contrastive learning

## FB15k-237 Dataset Details

The FB15k-237 dataset contains:
- **Source**: Subset of Freebase knowledge graph
- **Entities**: 14,541 (people, places, organizations, etc.)
- **Relations**: 237 types (born_in, works_for, capital_of, etc.)
- **Triples**: 310,116 total (train + valid + test)
- **Challenge**: Inverse relations removed to prevent trivial predictions
- **Task**: Predict missing head or tail entities given (h, r, ?) or (?, r, t)
- **Applications**: Question answering, information retrieval, knowledge base completion

## Improvement Ideas

1. **Quaternion Enhancements**:
   - Advanced quaternion normalization
   - Learnable quaternion constraints
   - Adaptive quaternion dimensionality
   - Quaternion attention mechanisms

2. **Distance-Adaptive Mechanisms**:
   - Learned distance metrics
   - Adaptive margin functions
   - Hierarchical distance modeling
   - Context-aware distance adaptation

3. **Rotation Improvements**:
   - Multi-scale rotations
   - Hierarchical rotation composition
   - Relation-specific rotation patterns
   - Symmetric rotation constraints

4. **Training Strategies**:
   - Self-adversarial negative sampling
   - Hard negative mining
   - Curriculum learning (simple to complex relations)
   - Multi-task learning with auxiliary objectives
   - Contrastive learning approaches

5. **Advanced Techniques**:
   - Graph attention for neighborhood information
   - Path-based reasoning (multi-hop)
   - Rule-based constraints
   - Temporal knowledge graph extensions
   - Uncertainty quantification

## References

- Paper: [Distance-Adaptive Quaternion Knowledge Graph Embedding with Bidirectional Rotation](https://arxiv.org/abs/2412.04076v2)
- FB15k-237: https://www.microsoft.com/en-us/download/details.aspx?id=52312
- Quaternions: 4D extension of complex numbers for 3D rotations
- Knowledge Graph Embeddings: Survey at https://arxiv.org/abs/2002.00819