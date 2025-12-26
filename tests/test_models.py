"""
Tests for model components.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vision import DummyVisionEncoder, create_vision_encoder
from models.reasoning import DummyTextReasoner, create_text_reasoner
from models.projection import ProjectionLayer, MultiTokenProjection
from models.multimodal import create_multimodal_vqa


class TestVisionEncoder:
    """Tests for vision encoder."""
    
    def test_dummy_encoder(self):
        """Test dummy vision encoder."""
        encoder = DummyVisionEncoder(embedding_dim=512)
        
        images = torch.randn(2, 3, 224, 224)
        embeddings = encoder(images)
        
        assert embeddings.shape == (2, 512)
    
    def test_embedding_normalization(self):
        """Test that embeddings are normalized."""
        encoder = DummyVisionEncoder(embedding_dim=512)
        
        images = torch.randn(2, 3, 224, 224)
        embeddings = encoder(images)
        
        norms = embeddings.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=0.01)
    
    def test_frozen_verification(self):
        """Test frozen parameter verification."""
        encoder = DummyVisionEncoder()
        assert encoder.verify_frozen() is True


class TestTextReasoner:
    """Tests for text reasoner."""
    
    def test_dummy_reasoner(self):
        """Test dummy text reasoner."""
        reasoner = DummyTextReasoner()
        
        tokens = reasoner.tokenize(["What color is the cube?"])
        output = reasoner.generate_greedy(
            tokens["input_ids"],
            tokens["attention_mask"],
        )
        
        assert len(output.texts) == 1
    
    def test_sampling_generation(self):
        """Test sampling generation returns log probs."""
        reasoner = DummyTextReasoner()
        
        tokens = reasoner.tokenize(["What color is the cube?"])
        output = reasoner.generate_sample(
            tokens["input_ids"],
            tokens["attention_mask"],
        )
        
        assert output.log_probs is not None
        assert output.entropy is not None


class TestProjectionLayer:
    """Tests for projection layer."""
    
    def test_simple_projection(self):
        """Test simple linear projection."""
        proj = ProjectionLayer(input_dim=512, output_dim=512)
        
        x = torch.randn(4, 512)
        y = proj(x)
        
        assert y.shape == (4, 512)
    
    def test_hidden_projection(self):
        """Test projection with hidden layer."""
        proj = ProjectionLayer(
            input_dim=512,
            output_dim=512,
            hidden_dim=256,
            use_hidden=True,
        )
        
        x = torch.randn(4, 512)
        y = proj(x)
        
        assert y.shape == (4, 512)
    
    def test_multi_token_projection(self):
        """Test multi-token projection."""
        proj = MultiTokenProjection(
            input_dim=512,
            output_dim=512,
            num_tokens=4,
        )
        
        x = torch.randn(2, 512)
        y = proj(x)
        
        assert y.shape == (2, 4, 512)
    
    def test_gradient_flow(self):
        """Test that gradients flow through projection."""
        proj = ProjectionLayer(input_dim=512, output_dim=512)
        
        x = torch.randn(4, 512, requires_grad=True)
        y = proj(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None


class TestMultimodalVQA:
    """Tests for the full multimodal model."""
    
    def test_creation(self):
        """Test model creation with dummy components."""
        model = create_multimodal_vqa(use_dummy=True)
        
        assert model is not None
    
    def test_forward_greedy(self):
        """Test greedy forward pass."""
        model = create_multimodal_vqa(use_dummy=True)
        
        images = torch.randn(2, 3, 224, 224)
        questions = ["What color?", "What shape?"]
        
        output = model(images, questions, mode="greedy")
        
        assert len(output.answers) == 2
    
    def test_forward_sample(self):
        """Test sampling forward pass."""
        model = create_multimodal_vqa(use_dummy=True)
        
        images = torch.randn(2, 3, 224, 224)
        questions = ["What color?", "What shape?"]
        
        output = model(images, questions, mode="sample")
        
        assert output.log_probs is not None
    
    def test_parameter_counts(self):
        """Test parameter counting."""
        model = create_multimodal_vqa(use_dummy=True)
        
        counts = model.count_parameters()
        
        assert "vision_encoder" in counts
        assert "projection_layer" in counts
        assert "text_reasoner" in counts
        assert "trainable" in counts
    
    def test_trainable_parameters(self):
        """Test getting trainable parameters."""
        model = create_multimodal_vqa(use_dummy=True)
        
        params = list(model.get_trainable_parameters())
        
        assert len(params) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
