#!/usr/bin/env python3
"""
Test script to demonstrate the new evaluation metrics
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

def test_metrics():
    """Test the evaluation metrics with sample data"""
    
    # Sample data (4 classes: real, deepfake, face2face, neuraltextures)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample predictions and true labels
    true_labels = np.random.randint(0, 4, n_samples)
    predictions = np.random.randint(0, 4, n_samples)
    
    # Generate sample probabilities (4 classes)
    probabilities = np.random.rand(n_samples, 4)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)  # Normalize
    
    # Calculate metrics
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    f1_per_class = f1_score(true_labels, predictions, average=None)
    
    try:
        auc_roc = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='macro')
    except ValueError as e:
        print(f"Could not calculate AUC-ROC: {e}")
        auc_roc = None
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Class names
    class_names = ['real', 'deepfake', 'face2face', 'neuraltextures']
    
    # Display results
    print("=" * 60)
    print("SAMPLE EVALUATION METRICS")
    print("=" * 60)
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC (Macro): {auc_roc:.4f}")
    else:
        print("AUC-ROC: Could not calculate")
    
    print("\nPer-Class F1 Scores:")
    for i, (class_name, f1) in enumerate(zip(class_names, f1_per_class)):
        print(f"  {class_name}: {f1:.4f}")
    
    print(f"\nConfusion Matrix:\n{cm}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))
    print("=" * 60)

if __name__ == "__main__":
    test_metrics()
