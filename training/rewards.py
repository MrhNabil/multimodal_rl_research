"""
Reward Functions for REINFORCE Training

Provides different reward signals for evaluating generated answers:
- Exact match: Binary reward for correct answers
- Partial match: Token overlap reward
- Length penalty: Penalize overly long responses
- Combined: Weighted combination of rewards
"""

import torch
from typing import List, Optional, Dict
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> torch.Tensor:
        """
        Compute rewards for predictions.
        
        Args:
            predictions: List of predicted answer strings
            targets: List of target answer strings
            
        Returns:
            Reward tensor [B]
        """
        pass


class ExactMatchReward(RewardFunction):
    """
    Exact match reward.
    
    Returns 1.0 if prediction matches target exactly (case-insensitive),
    0.0 otherwise.
    """
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize exact match reward.
        
        Args:
            case_sensitive: Whether to use case-sensitive matching
        """
        self.case_sensitive = case_sensitive
    
    def compute(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> torch.Tensor:
        """Compute exact match rewards."""
        rewards = []
        
        for pred, target in zip(predictions, targets):
            # Normalize
            pred_norm = pred.strip()
            target_norm = target.strip()
            
            if not self.case_sensitive:
                pred_norm = pred_norm.lower()
                target_norm = target_norm.lower()
            
            # Exact match
            reward = 1.0 if pred_norm == target_norm else 0.0
            rewards.append(reward)
        
        return torch.tensor(rewards)


class PartialMatchReward(RewardFunction):
    """
    Partial match reward based on token overlap.
    
    Computes F1-like score between prediction and target tokens.
    Returns value between 0.0 and 1.0.
    """
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize partial match reward.
        
        Args:
            case_sensitive: Whether to use case-sensitive matching
        """
        self.case_sensitive = case_sensitive
    
    def _tokenize(self, text: str) -> set:
        """Simple whitespace tokenization."""
        text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return set(text.split())
    
    def compute(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> torch.Tensor:
        """Compute partial match rewards."""
        rewards = []
        
        for pred, target in zip(predictions, targets):
            pred_tokens = self._tokenize(pred)
            target_tokens = self._tokenize(target)
            
            if not target_tokens:
                # Empty target: reward only if prediction is also empty
                reward = 1.0 if not pred_tokens else 0.0
            elif not pred_tokens:
                # Empty prediction: no reward
                reward = 0.0
            else:
                # Compute F1-like score
                overlap = len(pred_tokens & target_tokens)
                precision = overlap / len(pred_tokens)
                recall = overlap / len(target_tokens)
                
                if precision + recall > 0:
                    reward = 2 * precision * recall / (precision + recall)
                else:
                    reward = 0.0
            
            rewards.append(reward)
        
        return torch.tensor(rewards)


class LengthPenaltyReward(RewardFunction):
    """
    Length penalty reward.
    
    Penalizes responses that are longer than a target length.
    Can be combined with other rewards.
    """
    
    def __init__(
        self,
        target_length: int = 5,
        penalty_per_token: float = 0.1,
        max_penalty: float = 0.5,
    ):
        """
        Initialize length penalty.
        
        Args:
            target_length: Target response length
            penalty_per_token: Penalty per extra token
            max_penalty: Maximum total penalty
        """
        self.target_length = target_length
        self.penalty_per_token = penalty_per_token
        self.max_penalty = max_penalty
    
    def compute(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> torch.Tensor:
        """Compute length penalties (as negative rewards)."""
        penalties = []
        
        for pred in predictions:
            num_tokens = len(pred.strip().split())
            
            if num_tokens > self.target_length:
                # Apply penalty for extra tokens
                extra = num_tokens - self.target_length
                penalty = min(extra * self.penalty_per_token, self.max_penalty)
            else:
                penalty = 0.0
            
            penalties.append(-penalty)  # Negative penalty
        
        return torch.tensor(penalties)


class CombinedReward(RewardFunction):
    """
    Combination of multiple reward functions.
    
    Computes weighted sum of individual rewards.
    """
    
    def __init__(
        self,
        rewards: List[RewardFunction],
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize combined reward.
        
        Args:
            rewards: List of reward functions
            weights: Weights for each reward (default: equal weights)
        """
        self.rewards = rewards
        self.weights = weights or [1.0] * len(rewards)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def compute(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> torch.Tensor:
        """Compute combined rewards."""
        combined = torch.zeros(len(predictions))
        
        for reward_fn, weight in zip(self.rewards, self.weights):
            combined += weight * reward_fn.compute(predictions, targets)
        
        return combined


class ProgressiveReward(RewardFunction):
    """
    Progressive reward that becomes stricter over training.
    
    Starts with partial match, transitions to exact match.
    """
    
    def __init__(
        self,
        exact_match: ExactMatchReward = None,
        partial_match: PartialMatchReward = None,
        transition_steps: int = 1000,
    ):
        """
        Initialize progressive reward.
        
        Args:
            exact_match: Exact match reward function
            partial_match: Partial match reward function
            transition_steps: Number of steps for full transition
        """
        self.exact_match = exact_match or ExactMatchReward()
        self.partial_match = partial_match or PartialMatchReward()
        self.transition_steps = transition_steps
        self.current_step = 0
    
    def step(self):
        """Advance one step."""
        self.current_step += 1
    
    def set_step(self, step: int):
        """Set current step."""
        self.current_step = step
    
    def compute(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> torch.Tensor:
        """Compute progressive reward."""
        # Compute mixing weight (0 = all partial, 1 = all exact)
        alpha = min(self.current_step / self.transition_steps, 1.0)
        
        exact_rewards = self.exact_match.compute(predictions, targets)
        partial_rewards = self.partial_match.compute(predictions, targets)
        
        return alpha * exact_rewards + (1 - alpha) * partial_rewards


def create_reward_function(
    reward_type: str = "exact_match",
    **kwargs,
) -> RewardFunction:
    """
    Factory function to create reward functions.
    
    Args:
        reward_type: Type of reward ("exact_match", "partial_match",
                    "length_penalty", "combined", "progressive")
        **kwargs: Additional arguments for the reward function
        
    Returns:
        RewardFunction instance
    """
    if reward_type == "exact_match":
        return ExactMatchReward(
            case_sensitive=kwargs.get("case_sensitive", False)
        )
    elif reward_type == "partial_match":
        return PartialMatchReward(
            case_sensitive=kwargs.get("case_sensitive", False)
        )
    elif reward_type == "length_penalty":
        return LengthPenaltyReward(
            target_length=kwargs.get("target_length", 5),
            penalty_per_token=kwargs.get("penalty_per_token", 0.1),
            max_penalty=kwargs.get("max_penalty", 0.5),
        )
    elif reward_type == "combined":
        # Default: exact match + length penalty
        rewards = [
            ExactMatchReward(),
            LengthPenaltyReward(),
        ]
        weights = kwargs.get("weights", [0.8, 0.2])
        return CombinedReward(rewards=rewards, weights=weights)
    elif reward_type == "progressive":
        return ProgressiveReward(
            transition_steps=kwargs.get("transition_steps", 1000)
        )
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def main():
    """Test reward functions."""
    predictions = ["red", "blue", "red ball", ""]
    targets = ["red", "red", "red", "green"]
    
    # Exact match
    exact = ExactMatchReward()
    rewards = exact.compute(predictions, targets)
    print(f"Exact match rewards: {rewards.tolist()}")
    # Expected: [1.0, 0.0, 0.0, 0.0]
    
    # Partial match
    partial = PartialMatchReward()
    rewards = partial.compute(predictions, targets)
    print(f"Partial match rewards: {rewards.tolist()}")
    # Expected: [1.0, 0.0, 1.0, 0.0]
    
    # Length penalty
    length_pen = LengthPenaltyReward(target_length=1)
    rewards = length_pen.compute(predictions, targets)
    print(f"Length penalty rewards: {rewards.tolist()}")
    
    # Combined
    combined = create_reward_function("combined")
    rewards = combined.compute(predictions, targets)
    print(f"Combined rewards: {rewards.tolist()}")
    
    # Progressive
    progressive = ProgressiveReward(transition_steps=100)
    for step in [0, 50, 100]:
        progressive.set_step(step)
        rewards = progressive.compute(predictions, targets)
        print(f"Progressive rewards (step {step}): {rewards.tolist()}")


if __name__ == "__main__":
    main()
