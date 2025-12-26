"""
Tests for dataset generation and loading.
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_clevr import SyntheticCLEVRGenerator, Sample
from data.dataset import MultimodalDataset


class TestSyntheticCLEVRGenerator:
    """Tests for the synthetic CLEVR generator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SyntheticCLEVRGenerator(
                output_dir=tmpdir,
                image_size=224,
                seed=42,
            )
            
            assert generator.image_size == 224
            assert generator.seed == 42
    
    def test_small_dataset_generation(self):
        """Test generating a small dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SyntheticCLEVRGenerator(
                output_dir=tmpdir,
                image_size=224,
                seed=42,
            )
            
            datasets = generator.generate_dataset(
                num_train=10,
                num_val=5,
                num_test=5,
            )
            
            assert len(datasets["train"]) == 10
            assert len(datasets["val"]) == 5
            assert len(datasets["test"]) == 5
    
    def test_sample_structure(self):
        """Test that samples have correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SyntheticCLEVRGenerator(
                output_dir=tmpdir,
                seed=42,
            )
            
            datasets = generator.generate_dataset(num_train=5, num_val=1, num_test=1)
            sample = datasets["train"][0]
            
            assert isinstance(sample, Sample)
            assert sample.question is not None
            assert sample.answer is not None
            assert sample.question_type in ["color", "shape", "count", "spatial"]
    
    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                gen1 = SyntheticCLEVRGenerator(output_dir=tmpdir1, seed=42)
                gen2 = SyntheticCLEVRGenerator(output_dir=tmpdir2, seed=42)
                
                ds1 = gen1.generate_dataset(num_train=5, num_val=1, num_test=1)
                ds2 = gen2.generate_dataset(num_train=5, num_val=1, num_test=1)
                
                assert ds1["train"][0].question == ds2["train"][0].question
                assert ds1["train"][0].answer == ds2["train"][0].answer


class TestMultimodalDataset:
    """Tests for the PyTorch dataset."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        tmpdir = tempfile.mkdtemp()
        
        generator = SyntheticCLEVRGenerator(
            output_dir=tmpdir,
            seed=42,
        )
        generator.generate_dataset(num_train=10, num_val=5, num_test=5)
        
        return tmpdir
    
    def test_dataset_loading(self, sample_data):
        """Test loading dataset from disk."""
        dataset = MultimodalDataset(
            data_dir=sample_data,
            split="train",
        )
        
        assert len(dataset) == 10
    
    def test_getitem(self, sample_data):
        """Test getting individual items."""
        dataset = MultimodalDataset(
            data_dir=sample_data,
            split="train",
        )
        
        item = dataset[0]
        
        assert "image" in item
        assert "question" in item
        assert "answer" in item
        assert "question_type" in item
    
    def test_filter_by_question_type(self, sample_data):
        """Test filtering by question type."""
        dataset = MultimodalDataset(
            data_dir=sample_data,
            split="train",
            question_types=["color"],
        )
        
        for i in range(min(len(dataset), 5)):
            item = dataset[i]
            assert item["question_type"] == "color"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
