#!/usr/bin/env python3
"""
Solution template for CNN/Daily Mail SigExt summarization task

Agent should implement the evaluate() function to train and evaluate their model


**Technical Requirements for Validation:**

Your solution must save these artifacts for automatic validation:

1. Model checkpoint: 'model.pt' (≥5.0 MB)
   ```python
   torch.save(model.state_dict(), 'model.pt')
   ```

2. Training log: 'training.json' with:
   - batches_seen (≥50)
   - train_loss_history (must show ≥0.05 drop)
   - final_metrics (must match your return values exactly)
   - gpu_used field
   
   ```python
   result = evaluate_model()  # Get actual results
   log = {
       "batches_seen": 100,
       "train_loss_history": loss_history,
       "final_metrics": result,  # Must match returned values!
       "gpu_used": "cuda:0"
   }
   json.dump(log, open('training.json', 'w'))
   return result
   ```

See training_log_example.json for full format.
⚠️  IMPORTANT: Do NOT copy training_log_example.json - we will detect it.
⚠️  Advanced validation will catch any attempts to fabricate or shortcut results.
"""

from typing import Dict
import pathlib
import json


def evaluate() -> Dict[str, float]:
    """
    Train and evaluate a summarization model on the CNN/Daily Mail dataset
    
    This is a baseline template. The actual SigExt implementation would:
    1. Extract keyphrases from source documents
    2. Use these keyphrases to guide LLM-based summarization
    3. Generate summaries and evaluate with ROUGE metrics
    
    Returns:
        dict: Dictionary containing evaluation metrics
              - 'rouge_l': ROUGE-L F1 score (0-100)
    """
    
    # TODO: Implement your improved SigExt model here
    # 
    # Suggested approach:
    # 1. Load the CNN/Daily Mail dataset from /home/agent/dataset/
    # 2. Implement keyphrase extraction (or improve existing method)
    # 3. Design effective prompts using extracted keyphrases
    # 4. Use an LLM to generate summaries
    # 5. Evaluate using ROUGE-L F1 score
    #
    # The baseline SigExt achieves ROUGE-L of 27.4
    # Your goal is to exceed this score
    
    # Baseline implementation: return the documented baseline score
    # Replace this with your actual implementation!
    try:
        import os
        
        # Check if dataset exists
        dataset_path = pathlib.Path("/home/agent/dataset")
        if not dataset_path.exists():
            print("Warning: Dataset not found. Returning baseline value.")
            return {'rouge_l': 27.4}
        
        # Check if the SigExt repository is available
        repo_path = pathlib.Path("/home/agent/solution/src")
        if not repo_path.exists():
            print("Warning: SigExt repository not found. Returning baseline value.")
            return {'rouge_l': 27.4}
        
        print("Dataset and repository found.")
        print("NOTE: This is a placeholder baseline. Implement your SigExt improvements!")
        print("The actual SigExt model achieves ROUGE-L of 27.4.")
        print("You should:")
        print("  1. Improve keyphrase extraction quality")
        print("  2. Optimize prompt design for LLM summarization")
        print("  3. Experiment with different LLMs or prompting strategies")
        print("  4. Fine-tune the model on the CNN/Daily Mail dataset")
        
        # TODO: Replace this with actual model training and evaluation
        # Example workflow:
        # - Load dataset: dataset = load_dataset('cnn_dailymail', '3.0.0')
        # - Extract keyphrases from documents
        # - Generate summaries using LLM with keyphrase-guided prompts
        # - Compute ROUGE-L score: rouge_l = compute_rouge_l(predictions, references)
        # - Return {'rouge_l': rouge_l * 100}  # Convert to 0-100 scale
        
    except Exception as e:
        print(f"Error in baseline implementation: {e}")
        print("Returning documented baseline value.")
    
    # Return the baseline ROUGE-L score from the paper
    return {'rouge_l': 27.4}


if __name__ == "__main__":
    # Test the evaluate function
    result = evaluate()
    print(f"\nROUGE-L: {result['rouge_l']:.2f}")

