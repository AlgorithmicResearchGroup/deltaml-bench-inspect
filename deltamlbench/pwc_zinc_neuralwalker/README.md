# ZINC Molecular Property Prediction with NeuralWalker

## Overview
Improve NeuralWalker for predicting molecular properties on ZINC dataset. NeuralWalker uses random walks + sequence models to capture long-range dependencies in molecular graphs.

## Paper
**Title:** Learning Long Range Dependencies on Graphs via Random Walks  
**URL:** https://arxiv.org/abs/2406.03386v2  
**Venue:** ICLR 2025

## Baseline
- **MAE:** 0.065 ± 0.001 (lower is better!)

## Task
Train on 12,000 molecular graphs, predict constrained solubility. **Lower MAE = better performance.**

## Scoring
```
Score = (baseline_mae - your_mae) / baseline_mae × 100
```
**Examples:**
- MAE 0.060 → 7.69% improvement ✅
- MAE 0.063 → 3.08% improvement
- MAE 0.070 → 0% (worse than baseline)

## NeuralWalker Architecture

### Components
1. **Random Walk Sampler:** Samples m walks per node
2. **Walk Encoder:** Embeds node/edge features
3. **Sequence Layer:** Processes walks (CNN/RNN/Transformer/Mamba)
4. **Walk Aggregator:** Pools walk features to nodes
5. **Message Passing:** Optional local/global info exchange

### Why It Works
- Random walks capture long-range molecular interactions
- Sequence models process walk patterns effectively
- Flexible: works with various sequence architectures

## Dataset: ZINC
- **12,000 training molecular graphs**
- **Node features:** Atom type, chirality, hybridization
- **Edge features:** Bond type, conjugation, rings
- **Target:** Constrained solubility (continuous)
- **Available via:** PyTorch Geometric

## Improvement Strategies
1. **Walk Parameters:** Optimize walk length, sampling rate
2. **Sequence Model:** Try Transformers, Mamba, or hybrids
3. **Features:** Better molecular descriptors (RDKit)
4. **Architecture:** Deeper models, skip connections
5. **Training:** Ensembles, better regularization

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_zinc_neuralwalker_main
```

## Training Time

| Training Time |
|---------------|
| 13 hours 54 minutes |

## Solution Interface
```python
def evaluate() -> Dict[str, float]:
    return {'mae': float}  # Lower is better!
```

## References
- [NeuralWalker Paper](https://arxiv.org/abs/2406.03386)
- [NeuralWalker GitHub](https://github.com/DexiongYung/NeuralWalker)
- [ZINC Dataset](https://www.dgl.ai/docs/api/python/dgl.data.html#zinc-dataset)

Good luck achieving MAE < 0.065!

