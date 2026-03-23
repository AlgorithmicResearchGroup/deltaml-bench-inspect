#!/usr/bin/env python3
"""
Solution template for kvasir_seg_emcad task
Agent should implement the evaluate() function to train and evaluate their improved EMCAD model


**Technical Requirements for Validation:**

Your solution must save these artifacts for automatic validation:

1. Model checkpoint: 'model.pt' (≥8.0 MB)
   ```python
   torch.save(model.state_dict(), 'model.pt')
   ```

2. Training log: 'training.json' with:
   - batches_seen (≥80)
   - train_loss_history (must show ≥0.08 drop)
   - final_metrics (must match your return values exactly)
   - gpu_used field
   
   ```python
   result = evaluate_model()  # Get actual results
   log = {
       "batches_seen": 120,
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

import numpy as np
import pathlib
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import torchvision.transforms as transforms
import json

def evaluate() -> Dict[str, float]:
    """
    Train and evaluate an improved EMCAD model on the Kvasir-SEG dataset
    
    Returns:
        dict: Dictionary containing evaluation metrics
              - 'mean_dice': mean Dice coefficient on test set
    """
    
    # SIMPLE BASELINE IMPLEMENTATION - Replace this with your improved EMCAD model!
    # This demonstrates the interface and provides a starting point
    
    # STEP 1: Load and preprocess the Kvasir-SEG dataset
    # Kvasir-SEG dataset contains:
    # - Endoscopic images of gastrointestinal tract
    # - Polyp segmentation masks for early cancer detection
    # - 1000 polyp images with corresponding segmentation masks
    # Dataset available at: /home/agent/dataset/
    
    try:
        import os
        import glob
        from PIL import Image
        
        dataset_path = pathlib.Path("/home/agent/dataset")
        
        # Check if dataset is available
        if not dataset_path.exists():
            print("Warning: Dataset not found. Returning baseline value.")
            return {'mean_dice': 0.928}
        
        # Try to find images and masks
        # Kvasir-SEG typically has 'images' and 'masks' folders
        images_dir = dataset_path / "images"
        masks_dir = dataset_path / "masks"
        
        if not images_dir.exists() or not masks_dir.exists():
            # Try alternative structures (e.g., after unzipping)
            potential_dirs = list(dataset_path.glob("**/images"))
            if potential_dirs:
                images_dir = potential_dirs[0]
                masks_dir = images_dir.parent / "masks"
        
        if not images_dir.exists() or not masks_dir.exists():
            print("Warning: Could not find images/masks directories. Returning baseline value.")
            return {'mean_dice': 0.928}
        
        # SIMPLE BASELINE: Calculate Dice on a subset using basic thresholding
        # This is NOT a good model - just demonstrates the interface!
        image_files = sorted(glob.glob(str(images_dir / "*.jpg")) + 
                           glob.glob(str(images_dir / "*.png")))
        
        if len(image_files) == 0:
            print("Warning: No images found. Returning baseline value.")
            return {'mean_dice': 0.928}
        
        # Evaluate on first 50 images as a quick test (or all if fewer)
        num_test = min(50, len(image_files))
        dice_scores = []
        
        for img_path in image_files[:num_test]:
            img_name = os.path.basename(img_path)
            mask_path = str(masks_dir / img_name)
            
            # Handle different extensions
            if not os.path.exists(mask_path):
                base_name = os.path.splitext(img_name)[0]
                for ext in ['.jpg', '.png']:
                    test_path = str(masks_dir / (base_name + ext))
                    if os.path.exists(test_path):
                        mask_path = test_path
                        break
            
            if not os.path.exists(mask_path):
                continue
            
            # Load image and mask
            image = np.array(Image.open(img_path).convert('RGB'))
            mask_gt = np.array(Image.open(mask_path).convert('L'))
            
            # SIMPLE BASELINE PREDICTION: 
            # Convert to grayscale, apply threshold, and basic morphology
            # This is intentionally simple - agents should replace with EMCAD!
            gray = np.mean(image, axis=2)
            # Simple threshold (this is a bad predictor, just for demo)
            pred_mask = (gray < 150).astype(np.uint8)  
            
            # Binarize ground truth
            mask_gt = (mask_gt > 128).astype(np.uint8)
            
            # Calculate Dice coefficient
            dice = calculate_dice_coefficient(pred_mask, mask_gt)
            dice_scores.append(dice)
        
        if len(dice_scores) > 0:
            mean_dice = float(np.mean(dice_scores))
            print(f"Simple baseline mean Dice on {len(dice_scores)} images: {mean_dice:.4f}")
            print("NOTE: This is a simple thresholding baseline, not EMCAD!")
            print("The actual EMCAD model achieves 0.928. Replace this with your implementation.")
            return {'mean_dice': mean_dice}
        
    except Exception as e:
        print(f"Error in baseline implementation: {e}")
        print("Returning documented baseline value.")
    
    # If anything fails, return the documented baseline
    # EMCAD baseline: mean Dice = 0.928 (92.8%)
    # TODO: Replace this entire function with your EMCAD model implementation!
    return {'mean_dice': 0.928}

def load_kvasir_dataset() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Load Kvasir-SEG medical segmentation dataset
    
    Returns:
        Tuple of (train_images, train_masks, test_images, test_masks)
    """
    try:
        dataset_path = pathlib.Path("/home/agent/dataset")
        
        # TODO: Implement medical dataset loading
        # Kvasir-SEG dataset structure:
        # - Medical endoscopic images (various formats)
        # - Corresponding segmentation masks for polyps
        # - Typically organized as:
        #   - images/ folder with polyp images
        #   - masks/ folder with corresponding segmentation masks
        
        train_images = []
        train_masks = []
        test_images = []
        test_masks = []
        
        # Example loading code:
        # images_dir = dataset_path / "images"
        # masks_dir = dataset_path / "masks"
        # 
        # all_images = sorted(list(images_dir.glob("*.jpg")))
        # Split 80/20 for train/test
        # split_idx = int(len(all_images) * 0.8)
        # 
        # for img_path in all_images[:split_idx]:
        #     img = cv2.imread(str(img_path))
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     train_images.append(img)
        #     
        #     mask_path = masks_dir / f"{img_path.stem}.jpg"
        #     mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        #     train_masks.append(mask)
        # 
        # Similar for test images...
        
        return train_images, train_masks, test_images, test_masks
        
    except Exception as e:
        print(f"Error loading medical dataset: {e}")
        return [], [], [], []

def preprocess_medical_image(image: np.ndarray, target_size: Tuple[int, int] = (352, 352)) -> torch.Tensor:
    """
    Preprocess medical image for EMCAD model
    
    Args:
        image: Input image as numpy array
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image as torch tensor
    """
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor

def calculate_dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Calculate Dice coefficient for segmentation evaluation
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth segmentation mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient (0 to 1, higher is better)
    """
    # Flatten the arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Calculate intersection and union
    intersection = np.sum(pred * target)
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    
    return dice

def load_emcad_model():
    """
    Load the EMCAD model from the provided codebase
    
    Returns:
        EMCAD model instance
    """
    # TODO: Implement model loading from /home/agent/solution_repo/
    # Example:
    # import sys
    # sys.path.append('/home/agent/solution_repo')
    # from models.emcad import EMCAD
    # model = EMCAD(num_classes=1)
    # return model
    pass

def train_model(model, train_images: List[np.ndarray], train_masks: List[np.ndarray], 
                epochs: int = 100, batch_size: int = 8) -> nn.Module:
    """
    Train the EMCAD model
    
    Args:
        model: EMCAD model instance
        train_images: Training images
        train_masks: Training masks
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model
    """
    # TODO: Implement training logic
    # - Create DataLoader
    # - Define optimizer (Adam, SGD, etc.)
    # - Define loss function (Dice loss, BCE loss, combined loss)
    # - Training loop with validation
    # - Learning rate scheduling
    # - Model checkpointing
    pass

def evaluate_dice(model, test_images: List[np.ndarray], test_masks: List[np.ndarray]) -> float:
    """
    Evaluate the model and return mean Dice coefficient
    
    Args:
        model: Trained EMCAD model
        test_images: Test images
        test_masks: Test masks
        
    Returns:
        Mean Dice coefficient across all test images
    """
    # TODO: Implement evaluation logic
    # - Put model in evaluation mode
    # - Run inference on test images
    # - Calculate Dice coefficient for each image
    # - Return mean Dice
    
    dice_scores = []
    # for img, mask in zip(test_images, test_masks):
    #     pred_mask = model_predict(model, img)
    #     dice = calculate_dice_coefficient(pred_mask, mask)
    #     dice_scores.append(dice)
    # 
    # mean_dice = np.mean(dice_scores)
    # return mean_dice
    pass

if __name__ == "__main__":
    # Test the evaluate function
    result = evaluate()
    print(f"mean Dice: {result['mean_dice']:.4f}")
    
    # Compare with baseline
    baseline_value = 0.928
    
    improvement = (result['mean_dice'] - baseline_value) / baseline_value * 100
    
    print(f"\nmean Dice improvement: {improvement:.2f}%")