"""
Evaluation Metrics for Multimodal VQA

Provides metrics for evaluating model performance:
- Accuracy (exact match)
- Per-question-type accuracy
- Skill retention (CLIP embedding quality)
"""

import torch
from typing import Dict, List, Tuple, Optional
import numpy as np


def compute_accuracy(
    predictions: List[str],
    targets: List[str],
    case_sensitive: bool = False,
) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: List of predicted answers
        targets: List of target answers
        case_sensitive: Whether to use case-sensitive matching
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if not predictions:
        return 0.0
    
    correct = 0
    for pred, target in zip(predictions, targets):
        pred_norm = pred.strip()
        target_norm = target.strip()
        
        if not case_sensitive:
            pred_norm = pred_norm.lower()
            target_norm = target_norm.lower()
        
        if pred_norm == target_norm:
            correct += 1
    
    return correct / len(predictions)


def compute_per_type_accuracy(
    predictions: List[str],
    targets: List[str],
    question_types: List[str],
    case_sensitive: bool = False,
) -> Dict[str, float]:
    """
    Compute accuracy for each question type.
    
    Args:
        predictions: List of predicted answers
        targets: List of target answers
        question_types: List of question types
        case_sensitive: Whether to use case-sensitive matching
        
    Returns:
        Dictionary mapping question type to accuracy
    """
    # Group by question type
    type_correct = {}
    type_total = {}
    
    for pred, target, q_type in zip(predictions, targets, question_types):
        if q_type not in type_correct:
            type_correct[q_type] = 0
            type_total[q_type] = 0
        
        type_total[q_type] += 1
        
        pred_norm = pred.strip()
        target_norm = target.strip()
        
        if not case_sensitive:
            pred_norm = pred_norm.lower()
            target_norm = target_norm.lower()
        
        if pred_norm == target_norm:
            type_correct[q_type] += 1
    
    # Compute accuracy per type
    return {
        q_type: type_correct[q_type] / type_total[q_type]
        for q_type in type_correct
    }


def compute_skill_retention(
    original_embeddings: torch.Tensor,
    current_embeddings: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute skill retention metrics.
    
    Measures how well the vision encoder's representations
    are preserved after RL training.
    
    Args:
        original_embeddings: Embeddings from frozen CLIP [N, D]
        current_embeddings: Embeddings after training [N, D]
        
    Returns:
        Dictionary with retention metrics
    """
    # Cosine similarity
    original_norm = original_embeddings / original_embeddings.norm(dim=-1, keepdim=True)
    current_norm = current_embeddings / current_embeddings.norm(dim=-1, keepdim=True)
    
    cosine_sim = (original_norm * current_norm).sum(dim=-1)
    
    # Mean squared error
    mse = ((original_embeddings - current_embeddings) ** 2).mean(dim=-1)
    
    return {
        "cosine_similarity": cosine_sim.mean().item(),
        "cosine_similarity_std": cosine_sim.std().item(),
        "mse": mse.mean().item(),
        "mse_std": mse.std().item(),
    }


def compute_token_overlap(
    predictions: List[str],
    targets: List[str],
) -> Dict[str, float]:
    """
    Compute token-level overlap metrics.
    
    Args:
        predictions: List of predicted answers
        targets: List of target answers
        
    Returns:
        Dictionary with precision, recall, and F1
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    for pred, target in zip(predictions, targets):
        pred_tokens = set(pred.lower().strip().split())
        target_tokens = set(target.lower().strip().split())
        
        if not target_tokens:
            continue
        
        overlap = len(pred_tokens & target_tokens)
        
        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(target_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    n = len(predictions)
    return {
        "precision": total_precision / n if n > 0 else 0.0,
        "recall": total_recall / n if n > 0 else 0.0,
        "f1": total_f1 / n if n > 0 else 0.0,
    }


def compute_answer_distribution(
    predictions: List[str],
) -> Dict[str, int]:
    """
    Compute distribution of predicted answers.
    
    Useful for detecting mode collapse.
    
    Args:
        predictions: List of predicted answers
        
    Returns:
        Dictionary mapping answer to count
    """
    distribution = {}
    for pred in predictions:
        pred_norm = pred.lower().strip()
        distribution[pred_norm] = distribution.get(pred_norm, 0) + 1
    return distribution


def compute_confidence_calibration(
    predictions: List[str],
    targets: List[str],
    log_probs: torch.Tensor,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute confidence calibration metrics.
    
    Measures how well the model's confidence matches its accuracy.
    
    Args:
        predictions: List of predicted answers
        targets: List of target answers
        log_probs: Log probabilities [B, L]
        num_bins: Number of calibration bins
        
    Returns:
        Dictionary with calibration metrics
    """
    # Compute confidence as mean log prob
    confidences = log_probs.mean(dim=-1).exp().cpu().numpy()
    
    # Compute correctness
    correct = np.array([
        pred.lower().strip() == target.lower().strip()
        for pred, target in zip(predictions, targets)
    ])
    
    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0  # Expected calibration error
    
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_accuracy = correct[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            bin_size = in_bin.sum() / len(confidences)
            ece += bin_size * np.abs(bin_accuracy - bin_confidence)
    
    return {
        "expected_calibration_error": ece,
        "mean_confidence": confidences.mean(),
        "mean_accuracy": correct.mean(),
    }


def main():
    """Test evaluation metrics."""
    predictions = ["red", "blue", "red ball", "green"]
    targets = ["red", "red", "red", "green"]
    question_types = ["color", "color", "color", "color"]
    
    # Accuracy
    acc = compute_accuracy(predictions, targets)
    print(f"Accuracy: {acc:.4f}")
    
    # Per-type accuracy
    per_type = compute_per_type_accuracy(predictions, targets, question_types)
    print(f"Per-type accuracy: {per_type}")
    
    # Token overlap
    overlap = compute_token_overlap(predictions, targets)
    print(f"Token overlap: {overlap}")
    
    # Answer distribution
    dist = compute_answer_distribution(predictions)
    print(f"Answer distribution: {dist}")


if __name__ == "__main__":
    main()
