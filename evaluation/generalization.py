"""
Generalization Evaluator

Tests model's ability to generalize to:
- Unseen attribute combinations
- Novel question formulations
- Out-of-distribution samples
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from tqdm import tqdm

from .metrics import compute_accuracy, compute_per_type_accuracy


@dataclass
class GeneralizationResults:
    """Results from generalization evaluation."""
    seen_accuracy: float
    unseen_accuracy: float
    generalization_gap: float
    seen_per_type: Dict[str, float]
    unseen_per_type: Dict[str, float]
    num_seen: int
    num_unseen: int


class GeneralizationEvaluator:
    """
    Evaluates model generalization to unseen compositions.
    
    Tests whether the model can answer questions about
    attribute combinations not seen during training.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        holdout_combinations: List[Tuple[str, str]],
        device: str = "cpu",
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the generalization evaluator.
        
        Args:
            model: MultimodalVQA model
            holdout_combinations: List of (color, shape) pairs held out
            device: Device to use
            output_dir: Directory to save results
        """
        self.model = model
        self.holdout_combinations = set(holdout_combinations)
        self.device = device
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def is_unseen(self, objects: List[Dict]) -> bool:
        """
        Check if a sample contains unseen attribute combinations.
        
        Args:
            objects: List of object dictionaries with 'color' and 'shape'
            
        Returns:
            True if any object has a held-out combination
        """
        for obj in objects:
            combo = (obj.get("color", ""), obj.get("shape", ""))
            if combo in self.holdout_combinations:
                return True
        return False
    
    def evaluate(
        self,
        dataloader,
        split_name: str = "test",
    ) -> GeneralizationResults:
        """
        Evaluate generalization to unseen combinations.
        
        Args:
            dataloader: DataLoader with 'objects' field in samples
            split_name: Name of the split
            
        Returns:
            GeneralizationResults
        """
        self.model.eval()
        
        # Separate seen and unseen samples
        seen_predictions = []
        seen_targets = []
        seen_types = []
        
        unseen_predictions = []
        unseen_targets = []
        unseen_types = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generalization eval"):
                images = batch.images.to(self.device)
                questions = batch.questions
                targets = batch.answers
                question_types = batch.question_types
                
                # Generate predictions
                output = self.model(images, questions, mode="greedy")
                
                # Note: We need objects info from the dataset
                # For now, treat all as "seen" - in practice, the dataset
                # should provide this information
                seen_predictions.extend(output.answers)
                seen_targets.extend(targets)
                seen_types.extend(question_types)
        
        # Compute metrics
        seen_accuracy = compute_accuracy(seen_predictions, seen_targets)
        seen_per_type = compute_per_type_accuracy(
            seen_predictions, seen_targets, seen_types
        )
        
        # Unseen metrics (placeholder - need proper dataset support)
        unseen_accuracy = 0.0
        unseen_per_type = {}
        
        if unseen_predictions:
            unseen_accuracy = compute_accuracy(unseen_predictions, unseen_targets)
            unseen_per_type = compute_per_type_accuracy(
                unseen_predictions, unseen_targets, unseen_types
            )
        
        generalization_gap = seen_accuracy - unseen_accuracy
        
        results = GeneralizationResults(
            seen_accuracy=seen_accuracy,
            unseen_accuracy=unseen_accuracy,
            generalization_gap=generalization_gap,
            seen_per_type=seen_per_type,
            unseen_per_type=unseen_per_type,
            num_seen=len(seen_predictions),
            num_unseen=len(unseen_predictions),
        )
        
        # Save results
        if self.output_dir:
            self._save_results(results, split_name)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: GeneralizationResults, split_name: str):
        """Save generalization results."""
        results_path = os.path.join(
            self.output_dir,
            f"{split_name}_generalization.json"
        )
        with open(results_path, "w") as f:
            json.dump({
                "seen_accuracy": results.seen_accuracy,
                "unseen_accuracy": results.unseen_accuracy,
                "generalization_gap": results.generalization_gap,
                "seen_per_type": results.seen_per_type,
                "unseen_per_type": results.unseen_per_type,
                "num_seen": results.num_seen,
                "num_unseen": results.num_unseen,
            }, f, indent=2)
    
    def _print_summary(self, results: GeneralizationResults):
        """Print generalization summary."""
        print(f"\n{'='*50}")
        print("Generalization Results")
        print(f"{'='*50}")
        print(f"Seen accuracy: {results.seen_accuracy:.4f} ({results.num_seen} samples)")
        print(f"Unseen accuracy: {results.unseen_accuracy:.4f} ({results.num_unseen} samples)")
        print(f"Generalization gap: {results.generalization_gap:.4f}")
        
        print(f"\nSeen per-type accuracy:")
        for q_type, acc in sorted(results.seen_per_type.items()):
            print(f"  {q_type}: {acc:.4f}")
        
        if results.unseen_per_type:
            print(f"\nUnseen per-type accuracy:")
            for q_type, acc in sorted(results.unseen_per_type.items()):
                print(f"  {q_type}: {acc:.4f}")


def main():
    """Test the generalization evaluator."""
    from models.multimodal import create_multimodal_vqa
    
    print("Testing GeneralizationEvaluator...")
    
    model = create_multimodal_vqa(use_dummy=True)
    
    holdout = [("red", "cube"), ("blue", "sphere")]
    evaluator = GeneralizationEvaluator(
        model,
        holdout_combinations=holdout,
        output_dir="experiments/results/test_gen"
    )
    
    print(f"Holdout combinations: {holdout}")
    print("Evaluator created successfully!")


if __name__ == "__main__":
    main()
