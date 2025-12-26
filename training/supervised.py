"""
Supervised Trainer for Multimodal VQA

Provides a baseline using standard cross-entropy supervised learning.
Uses teacher forcing with ground truth answers.
"""

import os
import json
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm


@dataclass
class SupervisedConfig:
    """Configuration for supervised training."""
    batch_size: int = 32
    num_epochs: int = 10
    max_steps: int = 5000
    eval_every: int = 100
    save_every: int = 500
    log_every: int = 10
    
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    output_dir: str = "experiments/results"
    experiment_name: str = "supervised_training"


class SupervisedTrainer:
    """
    Supervised trainer for multimodal VQA.
    
    Uses cross-entropy loss with teacher forcing.
    Serves as a baseline to compare against RL training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SupervisedConfig,
        device: str = "cpu",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: MultimodalVQA model
            config: Training configuration
            device: Device to use
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.get_trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_accuracy = 0.0
        
        # Output
        os.makedirs(config.output_dir, exist_ok=True)
        self.run_dir = os.path.join(config.output_dir, config.experiment_name)
        os.makedirs(self.run_dir, exist_ok=True)
    
    def train_step(self, batch) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()
        
        images = batch.images.to(self.device)
        questions = batch.questions
        answers = batch.answers
        
        # Compute supervised loss
        loss = self.model.compute_supervised_loss(images, questions, answers)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        return {
            "step": self.global_step,
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.scheduler.get_last_lr()[0],
        }
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch.images.to(self.device)
                questions = batch.questions
                answers = batch.answers
                
                # Compute loss
                loss = self.model.compute_supervised_loss(images, questions, answers)
                total_loss += loss.item() * len(questions)
                
                # Generate answers
                output = self.model(images, questions, mode="greedy")
                
                # Compute accuracy
                for pred, target in zip(output.answers, answers):
                    if pred.strip().lower() == target.strip().lower():
                        total_correct += 1
                    total_samples += 1
        
        return {
            "accuracy": total_correct / max(total_samples, 1),
            "loss": total_loss / max(total_samples, 1),
            "num_samples": total_samples,
        }
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
    ) -> List[Dict]:
        """Train the model."""
        print("Starting supervised training...")
        
        history = []
        start_time = time.time()
        
        pbar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.global_step < self.config.max_steps:
            self.epoch += 1
            
            for batch in train_dataloader:
                if self.global_step >= self.config.max_steps:
                    break
                
                stats = self.train_step(batch)
                history.append(stats)
                
                pbar.update(1)
                pbar.set_postfix({"loss": f"{stats['loss']:.4f}"})
                
                # Logging
                if self.global_step % self.config.log_every == 0:
                    self._log_step(stats)
                
                # Evaluation
                if val_dataloader and self.global_step % self.config.eval_every == 0:
                    eval_metrics = self.evaluate(val_dataloader)
                    print(f"\nStep {self.global_step}: "
                          f"val_acc={eval_metrics['accuracy']:.4f}")
                    
                    if eval_metrics["accuracy"] > self.best_accuracy:
                        self.best_accuracy = eval_metrics["accuracy"]
                        self.save_checkpoint("best_model.pt")
                
                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
        
        pbar.close()
        self.save_checkpoint("final_model.pt")
        
        print(f"\nTraining complete! Best accuracy: {self.best_accuracy:.4f}")
        
        return history
    
    def _log_step(self, stats: Dict):
        """Log step to file."""
        log_path = os.path.join(self.run_dir, "training_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = os.path.join(self.run_dir, filename)
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_accuracy": self.best_accuracy,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_accuracy = checkpoint["best_accuracy"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])


def main():
    """Test the supervised trainer."""
    from models.multimodal import create_multimodal_vqa
    
    print("Testing SupervisedTrainer...")
    
    model = create_multimodal_vqa(use_dummy=True)
    config = SupervisedConfig(batch_size=4, max_steps=10)
    trainer = SupervisedTrainer(model, config)
    
    print("Trainer created successfully!")


if __name__ == "__main__":
    main()
