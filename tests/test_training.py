"""
Tests for training components.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.rewards import (
    ExactMatchReward,
    PartialMatchReward,
    LengthPenaltyReward,
    CombinedReward,
    create_reward_function,
)
from training.reinforce import MovingAverageBaseline


class TestRewardFunctions:
    """Tests for reward functions."""
    
    def test_exact_match_correct(self):
        """Test exact match with correct answers."""
        reward_fn = ExactMatchReward()
        
        preds = ["red", "blue", "green"]
        targets = ["red", "blue", "green"]
        
        rewards = reward_fn.compute(preds, targets)
        
        assert rewards.tolist() == [1.0, 1.0, 1.0]
    
    def test_exact_match_incorrect(self):
        """Test exact match with incorrect answers."""
        reward_fn = ExactMatchReward()
        
        preds = ["red", "blue", "green"]
        targets = ["blue", "green", "red"]
        
        rewards = reward_fn.compute(preds, targets)
        
        assert rewards.tolist() == [0.0, 0.0, 0.0]
    
    def test_exact_match_case_insensitive(self):
        """Test case insensitivity."""
        reward_fn = ExactMatchReward(case_sensitive=False)
        
        preds = ["RED", "Blue"]
        targets = ["red", "BLUE"]
        
        rewards = reward_fn.compute(preds, targets)
        
        assert rewards.tolist() == [1.0, 1.0]
    
    def test_partial_match(self):
        """Test partial match reward."""
        reward_fn = PartialMatchReward()
        
        preds = ["red ball", "blue"]
        targets = ["red", "blue"]
        
        rewards = reward_fn.compute(preds, targets)
        
        assert rewards[0] > 0  # Partial match
        assert rewards[1] == 1.0  # Exact match
    
    def test_length_penalty(self):
        """Test length penalty reward."""
        reward_fn = LengthPenaltyReward(target_length=1, penalty_per_token=0.1)
        
        preds = ["red", "red ball", "red ball cube"]
        targets = ["red", "red", "red"]
        
        rewards = reward_fn.compute(preds, targets)
        
        assert rewards[0] == 0.0  # No penalty
        assert rewards[1] < 0.0  # Penalty
        assert rewards[2] < rewards[1]  # More penalty
    
    def test_combined_reward(self):
        """Test combined reward function."""
        exact = ExactMatchReward()
        length = LengthPenaltyReward()
        
        combined = CombinedReward(
            rewards=[exact, length],
            weights=[0.8, 0.2],
        )
        
        preds = ["red"]
        targets = ["red"]
        
        rewards = combined.compute(preds, targets)
        
        assert len(rewards) == 1
    
    def test_create_reward_function(self):
        """Test factory function."""
        reward_fn = create_reward_function("exact_match")
        assert isinstance(reward_fn, ExactMatchReward)
        
        reward_fn = create_reward_function("partial_match")
        assert isinstance(reward_fn, PartialMatchReward)


class TestBaseline:
    """Tests for baseline functions."""
    
    def test_moving_average_initialization(self):
        """Test moving average baseline initialization."""
        baseline = MovingAverageBaseline(decay=0.99)
        
        assert baseline.value == 0.0
        assert not baseline.initialized
    
    def test_moving_average_update(self):
        """Test moving average update."""
        baseline = MovingAverageBaseline(decay=0.9)
        
        # First update initializes
        value = baseline.update(1.0)
        assert value == 1.0
        
        # Second update applies decay
        value = baseline.update(0.0)
        assert value == 0.9 * 1.0 + 0.1 * 0.0
    
    def test_moving_average_convergence(self):
        """Test that baseline converges to constant input."""
        baseline = MovingAverageBaseline(decay=0.9)
        
        for _ in range(100):
            value = baseline.update(0.5)
        
        assert abs(value - 0.5) < 0.01


class TestGradients:
    """Tests for gradient computation."""
    
    def test_policy_gradient_shape(self):
        """Test that policy gradient has correct shape."""
        from models.multimodal import create_multimodal_vqa
        
        model = create_multimodal_vqa(use_dummy=True)
        
        images = torch.randn(2, 3, 224, 224)
        questions = ["What color?", "What shape?"]
        
        output = model(images, questions, mode="sample")
        
        # Simulate policy gradient
        rewards = torch.tensor([1.0, 0.0])
        
        if output.log_probs is not None:
            log_probs_sum = output.log_probs.sum(dim=-1)
            loss = -(rewards * log_probs_sum).mean()
            
            assert loss.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
