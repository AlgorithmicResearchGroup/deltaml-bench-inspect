# CNN/Daily Mail Summarization with SigExt

## Overview
This task challenges agents to improve the SigExt (Salient Information Extraction) model for abstractive summarization on the CNN/Daily Mail dataset.

## Task Description
**Paper:** [Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization](https://www.amazon.science/publications/salient-information-prompting-to-steer-content-in-prompt-based-abstractive-summarization)

**Repository:** https://github.com/amazon-science/SigExt

**Dataset:** CNN/Daily Mail - A standard benchmark for news article summarization

## Baseline Performance
- **ROUGE-L F1:** 27.4

## How to Run

### Running with Inspect
```bash
# From the repo root
./scripts/bootstrap_inspect.sh
./run_benchmark.sh run pwc_cnn_main
```

## Training Time

| Training Time | 1 hour 40 minutes |

## Goal
Achieve higher ROUGE-L scores by improving the SigExt model's keyphrase extraction and prompt-based summarization approach.

## Evaluation Metric
**ROUGE-L F1 Score:** Measures the longest common subsequence between generated and reference summaries.

## Scoring Formula
```
Score = (your_rouge_l - baseline_rouge_l) / baseline_rouge_l × 100
```

### Examples
- **ROUGE-L 30.0** → Score: **9.5%** improvement
- **ROUGE-L 28.5** → Score: **4.0%** improvement  
- **ROUGE-L 27.4** → Score: **0%** (matches baseline)
- **ROUGE-L 25.0** → Score: **0%** (worse than baseline)

## Approach
The SigExt model works by:
1. Extracting salient keyphrases from source documents
2. Using these keyphrases to guide LLM prompts
3. Generating summaries that align better with reference texts

## Improvement Strategies
1. **Enhanced Keyphrase Extraction**
   - Improve keyphrase quality and relevance
   - Experiment with different extraction algorithms
   - Consider document structure and context

2. **Prompt Optimization**
   - Design more effective prompts for LLMs
   - Balance detail level and coherence
   - Incorporate keyphrases more strategically

3. **Model Architecture**
   - Fine-tune model components
   - Experiment with different LLM backbones
   - Optimize hyperparameters

4. **Training Strategy**
   - Better training procedures
   - Data augmentation techniques
   - Transfer learning from related tasks

## Files Structure
```
/home/agent/
├── paper.pdf          # Research paper
├── solution/          # SigExt codebase
│   └── solution.py    # Your implementation
├── dataset/           # CNN/Daily Mail dataset
└── score.py           # Scoring script
```

## Getting Started
1. Review the SigExt paper and codebase
2. Understand the keyphrase extraction process
3. Identify bottlenecks and improvement opportunities
4. Implement your enhancements in `solution/solution.py`
5. Test with the provided `evaluate()` function

## Solution Interface
Your `solution.py` must implement:

```python
def evaluate() -> Dict[str, float]:
    """
    Returns:
        {
            'rouge_l': float  # ROUGE-L F1 score (0-100)
        }
    """
```

## Common Pitfalls
- Not properly preprocessing the CNN/Daily Mail dataset
- Ineffective keyphrase extraction leading to poor prompts
- Suboptimal LLM selection or prompting strategy
- Overfitting to training data
- Computational efficiency issues with large-scale evaluation

## Resources
- **Paper:** `/home/agent/paper.pdf`
- **Code:** `/home/agent/solution/`
- **Dataset:** `/home/agent/dataset/`

## References
- [CNN/Daily Mail Dataset](https://huggingface.co/datasets/cnn_dailymail)
- [ROUGE Metric](https://en.wikipedia.org/wiki/ROUGE_(metric))
- [SigExt GitHub Repository](https://github.com/amazon-science/SigExt)

Good luck!

