"""
Evaluator for Multimodal VQA

Comprehensive evaluation pipeline that runs all metrics
on the VQA model and saves results.
"""

import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import torch
from tqdm import tqdm

from .metrics import (
    compute_accuracy,
    compute_per_type_accuracy,
    compute_skill_retention,
    compute_token_overlap,
    compute_answer_distribution,
)


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    accuracy: float
    per_type_accuracy: Dict[str, float]
    token_overlap: Dict[str, float]
    answer_distribution: Dict[str, int]
    skill_retention: Optional[Dict[str, float]] = None
    num_samples: int = 0
    evaluation_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class Evaluator:
    """
    Comprehensive evaluator for multimodal VQA.
    
    Runs all evaluation metrics and produces detailed reports.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: MultimodalVQA model to evaluate
            device: Device to use
            output_dir: Directory to save results
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(
        self,
        dataloader,
        split_name: str = "test",
        save_predictions: bool = True,
        reference_embeddings: Optional[torch.Tensor] = None,
    ) -> EvaluationResults:
        """
        Run full evaluation on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            split_name: Name of the split (for saving)
            save_predictions: Whether to save individual predictions
            reference_embeddings: Original CLIP embeddings for skill retention
            
        Returns:
            EvaluationResults with all metrics
        """
        self.model.eval()
        
        start_time = time.time()
        
        all_predictions = []
        all_targets = []
        all_question_types = []
        all_embeddings = []
        all_samples = []
        
        print(f"Evaluating on {split_name} split...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch.images.to(self.device)
                questions = batch.questions
                targets = batch.answers
                question_types = batch.question_types
                
                # Generate predictions
                output = self.model(images, questions, mode="greedy")
                
                # Collect results
                all_predictions.extend(output.answers)
                all_targets.extend(targets)
                all_question_types.extend(question_types)
                
                if output.image_embeddings is not None:
                    all_embeddings.append(output.image_embeddings.cpu())
                
                # Store samples for logging
                for i, (pred, target, q_type, q) in enumerate(
                    zip(output.answers, targets, question_types, questions)
                ):
                    all_samples.append({
                        "question": q,
                        "prediction": pred,
                        "target": target,
                        "question_type": q_type,
                        "correct": pred.lower().strip() == target.lower().strip(),
                    })
        
        # Compute metrics
        accuracy = compute_accuracy(all_predictions, all_targets)
        per_type_accuracy = compute_per_type_accuracy(
            all_predictions, all_targets, all_question_types
        )
        token_overlap = compute_token_overlap(all_predictions, all_targets)
        answer_dist = compute_answer_distribution(all_predictions)
        
        # Skill retention (if reference embeddings provided)
        skill_retention = None
        if reference_embeddings is not None and all_embeddings:
            current_embeddings = torch.cat(all_embeddings, dim=0)
            if current_embeddings.shape == reference_embeddings.shape:
                skill_retention = compute_skill_retention(
                    reference_embeddings, current_embeddings
                )
        
        evaluation_time = time.time() - start_time
        
        results = EvaluationResults(
            accuracy=accuracy,
            per_type_accuracy=per_type_accuracy,
            token_overlap=token_overlap,
            answer_distribution=answer_dist,
            skill_retention=skill_retention,
            num_samples=len(all_predictions),
            evaluation_time=evaluation_time,
        )
        
        # Save results
        if self.output_dir:
            self._save_results(results, all_samples, split_name, save_predictions)
        
        # Print summary
        self._print_summary(results, split_name)
        
        return results
    
    def _save_results(
        self,
        results: EvaluationResults,
        samples: List[Dict],
        split_name: str,
        save_predictions: bool,
    ):
        """Save evaluation results to files."""
        # Save metrics
        metrics_path = os.path.join(self.output_dir, f"{split_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        
        # Save predictions
        if save_predictions:
            predictions_path = os.path.join(self.output_dir, f"{split_name}_predictions.json")
            with open(predictions_path, "w") as f:
                json.dump(samples, f, indent=2)
    
    def _print_summary(self, results: EvaluationResults, split_name: str):
        """Print evaluation summary."""
        print(f"\n{'='*50}")
        print(f"Evaluation Results: {split_name}")
        print(f"{'='*50}")
        print(f"Overall accuracy: {results.accuracy:.4f}")
        print(f"Samples evaluated: {results.num_samples}")
        print(f"Time: {results.evaluation_time:.2f}s")
        
        print(f"\nPer-type accuracy:")
        for q_type, acc in sorted(results.per_type_accuracy.items()):
            print(f"  {q_type}: {acc:.4f}")
        
        print(f"\nToken overlap:")
        for metric, value in results.token_overlap.items():
            print(f"  {metric}: {value:.4f}")
        
        if results.skill_retention:
            print(f"\nSkill retention:")
            for metric, value in results.skill_retention.items():
                print(f"  {metric}: {value:.4f}")
        
        print(f"\nTop answer distribution:")
        sorted_dist = sorted(
            results.answer_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for answer, count in sorted_dist:
            print(f"  '{answer}': {count}")


def main():
    """Test the evaluator."""
    from models.multimodal import create_multimodal_vqa
    
    print("Testing Evaluator...")
    
    model = create_multimodal_vqa(use_dummy=True)
    evaluator = Evaluator(model, output_dir="experiments/results/test_eval")
    
    print("Evaluator created successfully!")


if __name__ == "__main__":
    main()
