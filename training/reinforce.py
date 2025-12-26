"""
REINFORCE Trainer for Multimodal VQA

Implements the REINFORCE algorithm (policy gradient) for training
the multimodal VQA model using reward signals instead of supervised labels.

Key features:
- Baseline variance reduction (moving average or learned)
- Entropy regularization for exploration
- Gradient clipping for stability
"""

import os
import json
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from tqdm import tqdm

from .rewards import RewardFunction, create_reward_function


@dataclass
class TrainingStats:
    """Statistics from training."""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    reward: float = 0.0
    baseline: float = 0.0
    entropy: float = 0.0
    grad_norm: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    time_elapsed: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for REINFORCE training."""
    # Training
    batch_size: int = 32
    num_epochs: int = 10
    max_steps: int = 5000
    eval_every: int = 100
    save_every: int = 500
    log_every: int = 10
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # REINFORCE
    reward_type: str = "exact_match"
    baseline_type: str = "moving_avg"  # "none", "moving_avg", "learned"
    baseline_decay: float = 0.99
    temperature: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # Output
    output_dir: str = "experiments/results"
    experiment_name: str = "rl_training"


class MovingAverageBaseline:
    """Moving average baseline for variance reduction."""
    
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.value = 0.0
        self.initialized = False
    
    def update(self, reward: float) -> float:
        """Update baseline and return current value."""
        if not self.initialized:
            self.value = reward
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1 - self.decay) * reward
        return self.value
    
    def get(self) -> float:
        """Get current baseline value."""
        return self.value


class LearnedBaseline(nn.Module):
    """Learned value function baseline."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class REINFORCETrainer:
    """
    REINFORCE trainer for multimodal VQA.
    
    Uses policy gradient to train the model to maximize
    expected reward on the VQA task.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        reward_fn: Optional[RewardFunction] = None,
        device: str = "cpu",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: MultimodalVQA model
            config: Training configuration
            reward_fn: Reward function (default: exact match)
            device: Device to use
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Reward function
        self.reward_fn = reward_fn or create_reward_function(config.reward_type)
        
        # Baseline
        if config.baseline_type == "moving_avg":
            self.baseline = MovingAverageBaseline(decay=config.baseline_decay)
        elif config.baseline_type == "learned":
            self.baseline = LearnedBaseline().to(device)
        else:
            self.baseline = None
        
        # Optimizer (only trainable parameters)
        trainable_params = list(model.get_trainable_parameters())
        if config.baseline_type == "learned":
            trainable_params.extend(self.baseline.parameters())
        
        self.optimizer = AdamW(
            trainable_params,
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
        self.training_history = []
        
        # Setup output directory
        os.makedirs(config.output_dir, exist_ok=True)
        self.run_dir = os.path.join(
            config.output_dir,
            config.experiment_name,
        )
        os.makedirs(self.run_dir, exist_ok=True)
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        baseline_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute REINFORCE policy gradient loss.
        
        Args:
            log_probs: Log probabilities of sampled actions [B, L]
            rewards: Reward values [B]
            baseline_value: Baseline for variance reduction
            
        Returns:
            Policy loss (negative of policy gradient objective)
        """
        # Sum log probs across sequence
        sequence_log_probs = log_probs.sum(dim=-1)
        
        # Compute advantage
        advantage = rewards - baseline_value
        
        # Policy gradient loss (maximize reward = minimize negative)
        policy_loss = -(advantage * sequence_log_probs).mean()
        
        return policy_loss
    
    def train_step(
        self,
        batch: Dict,
    ) -> TrainingStats:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data with images, questions, answers
            
        Returns:
            TrainingStats for this step
        """
        self.model.train()
        
        images = batch.images.to(self.device)
        questions = batch.questions
        target_answers = batch.answers
        
        # Generate answers using sampling
        output = self.model.generate(
            images,
            questions,
            temperature=self.config.temperature,
            return_log_probs=True,
        )
        
        # Compute rewards
        rewards = self.reward_fn.compute(output.answers, target_answers)
        rewards = rewards.to(self.device)
        
        # Update baseline
        mean_reward = rewards.mean().item()
        if isinstance(self.baseline, MovingAverageBaseline):
            baseline_value = self.baseline.update(mean_reward)
        elif isinstance(self.baseline, LearnedBaseline):
            baseline_value = self.baseline(output.image_embeddings).mean().item()
        else:
            baseline_value = 0.0
        
        # Compute policy loss
        policy_loss = self.compute_policy_loss(
            output.log_probs,
            rewards,
            baseline_value,
        )
        
        # Entropy regularization
        entropy_loss = 0.0
        if output.entropy is not None and self.config.entropy_coef > 0:
            entropy_loss = -self.config.entropy_coef * output.entropy.mean()
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute accuracy
        correct = sum(
            pred.strip().lower() == target.strip().lower()
            for pred, target in zip(output.answers, target_answers)
        )
        accuracy = correct / len(target_answers)
        
        self.global_step += 1
        
        return TrainingStats(
            step=self.global_step,
            epoch=self.epoch,
            loss=total_loss.item(),
            reward=mean_reward,
            baseline=baseline_value,
            entropy=output.entropy.mean().item() if output.entropy is not None else 0.0,
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            accuracy=accuracy,
            learning_rate=self.scheduler.get_last_lr()[0],
        )
    
    def evaluate(
        self,
        dataloader,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Evaluation dataloader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_correct = 0
        total_samples = 0
        total_reward = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch.images.to(self.device)
                questions = batch.questions
                target_answers = batch.answers
                
                # Generate answers (greedy for eval)
                output = self.model(
                    images,
                    questions,
                    mode="greedy",
                )
                
                # Compute accuracy
                for pred, target in zip(output.answers, target_answers):
                    if pred.strip().lower() == target.strip().lower():
                        total_correct += 1
                    total_samples += 1
                
                # Compute reward
                rewards = self.reward_fn.compute(output.answers, target_answers)
                total_reward += rewards.sum().item()
        
        return {
            "accuracy": total_correct / max(total_samples, 1),
            "reward": total_reward / max(total_samples, 1),
            "num_samples": total_samples,
        }
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        callback: Optional[Callable] = None,
    ) -> List[TrainingStats]:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            callback: Callback function called after each step
            
        Returns:
            List of training statistics
        """
        print(f"Starting REINFORCE training...")
        print(f"  Max steps: {self.config.max_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Reward type: {self.config.reward_type}")
        print(f"  Baseline type: {self.config.baseline_type}")
        
        history = []
        start_time = time.time()
        
        pbar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.global_step < self.config.max_steps:
            self.epoch += 1
            
            for batch in train_dataloader:
                if self.global_step >= self.config.max_steps:
                    break
                
                # Training step
                stats = self.train_step(batch)
                stats.time_elapsed = time.time() - start_time
                history.append(stats)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{stats.loss:.4f}",
                    "reward": f"{stats.reward:.4f}",
                    "acc": f"{stats.accuracy:.4f}",
                })
                
                # Logging
                if self.global_step % self.config.log_every == 0:
                    self._log_step(stats)
                
                # Evaluation
                if val_dataloader and self.global_step % self.config.eval_every == 0:
                    eval_metrics = self.evaluate(val_dataloader)
                    print(f"\nStep {self.global_step}: "
                          f"val_acc={eval_metrics['accuracy']:.4f}, "
                          f"val_reward={eval_metrics['reward']:.4f}")
                    
                    # Save best model
                    if eval_metrics["accuracy"] > self.best_accuracy:
                        self.best_accuracy = eval_metrics["accuracy"]
                        self.save_checkpoint("best_model.pt")
                
                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
                
                # Callback
                if callback:
                    callback(stats)
        
        pbar.close()
        
        # Final save
        self.save_checkpoint("final_model.pt")
        self._save_training_history(history)
        
        print(f"\nTraining complete!")
        print(f"  Total steps: {self.global_step}")
        print(f"  Best accuracy: {self.best_accuracy:.4f}")
        print(f"  Time elapsed: {time.time() - start_time:.1f}s")
        
        return history
    
    def _log_step(self, stats: TrainingStats):
        """Log training step to file."""
        log_path = os.path.join(self.run_dir, "training_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": stats.step,
                "epoch": stats.epoch,
                "loss": stats.loss,
                "reward": stats.reward,
                "baseline": stats.baseline,
                "entropy": stats.entropy,
                "accuracy": stats.accuracy,
                "lr": stats.learning_rate,
                "time": stats.time_elapsed,
            }) + "\n")
    
    def _save_training_history(self, history: List[TrainingStats]):
        """Save complete training history."""
        history_path = os.path.join(self.run_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump([{
                "step": s.step,
                "loss": s.loss,
                "reward": s.reward,
                "accuracy": s.accuracy,
            } for s in history], f, indent=2)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.run_dir, filename)
        torch.save({
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_accuracy": self.best_accuracy,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_accuracy = checkpoint["best_accuracy"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])


def main():
    """Test the REINFORCE trainer."""
    from models.multimodal import create_multimodal_vqa
    
    print("Testing REINFORCETrainer...")
    
    # Create dummy model
    model = create_multimodal_vqa(use_dummy=True)
    
    # Create trainer
    config = TrainingConfig(
        batch_size=4,
        max_steps=10,
        learning_rate=1e-4,
    )
    trainer = REINFORCETrainer(model, config)
    
    print(f"Trainer created with config:")
    print(f"  Reward type: {config.reward_type}")
    print(f"  Baseline type: {config.baseline_type}")


if __name__ == "__main__":
    main()
