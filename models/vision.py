"""
Vision Encoder Module (Frozen CLIP)

Wraps the CLIP vision encoder for extracting image embeddings.
The encoder is completely frozen - no gradients flow through it.

This represents "Atomic Skill A" in the research design.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from PIL import Image

# Use open_clip for flexibility
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("Warning: open_clip not available. Install with: pip install open-clip-torch")


class VisionEncoder(nn.Module):
    """
    Frozen CLIP vision encoder.
    
    Extracts 512-dimensional embeddings from images using CLIP's
    vision transformer. All parameters are frozen.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        embedding_dim: int = 512,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the vision encoder.
        
        Args:
            model_name: CLIP model architecture name
            pretrained: Pretrained weights source (e.g., "openai")
            embedding_dim: Output embedding dimension (512 for ViT-B-32)
            device: Device to use ("cpu" for this project)
            cache_dir: Directory to cache pretrained models
        """
        super().__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim
        self.device = device
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip is required. Install with: pip install open-clip-torch")
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name} ({pretrained})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            cache_dir=cache_dir,
        )
        
        # Freeze all parameters
        self._freeze_all_parameters()
        
        # Set to eval mode
        self.model.eval()
        
        print(f"CLIP model loaded. Embedding dim: {embedding_dim}")
    
    def _freeze_all_parameters(self) -> None:
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Double-check by counting frozen params
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        
        print(f"Frozen {frozen_params}/{total_params} parameters ({frozen_params/total_params*100:.1f}%)")
    
    def get_preprocess(self):
        """Return the preprocessing function for images."""
        return self.preprocess
    
    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image embeddings.
        
        Args:
            images: Batch of images [B, 3, H, W], preprocessed
            
        Returns:
            Image embeddings [B, embedding_dim]
        """
        # Encode images
        features = self.model.encode_image(images)
        
        # Normalize embeddings (CLIP convention)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    @torch.no_grad()
    def encode_pil_images(self, images: list) -> torch.Tensor:
        """
        Encode PIL images directly.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            Image embeddings [B, embedding_dim]
        """
        # Preprocess images
        processed = torch.stack([self.preprocess(img) for img in images])
        processed = processed.to(self.device)
        
        return self.forward(processed)
    
    def verify_frozen(self) -> bool:
        """Verify that all parameters are frozen."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Warning: Parameter {name} is not frozen!")
                return False
        return True
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


class DummyVisionEncoder(nn.Module):
    """
    Dummy vision encoder for testing without CLIP.
    
    Returns random embeddings. Useful for testing the pipeline
    without loading the full CLIP model.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        print("Using DummyVisionEncoder (random embeddings)")
    
    def get_preprocess(self):
        """Return a dummy preprocessing function."""
        def dummy_preprocess(image):
            # Convert PIL to tensor
            import numpy as np
            img_array = np.array(image.resize((224, 224)))
            return torch.tensor(img_array).permute(2, 0, 1).float() / 255.0
        return dummy_preprocess
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return random embeddings."""
        batch_size = images.shape[0]
        embeddings = torch.randn(batch_size, self.embedding_dim, device=self.device)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    
    def verify_frozen(self) -> bool:
        """Dummy is always frozen."""
        return True
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


def create_vision_encoder(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    embedding_dim: int = 512,
    device: str = "cpu",
    use_dummy: bool = False,
    cache_dir: Optional[str] = None,
) -> nn.Module:
    """
    Factory function to create a vision encoder.
    
    Args:
        model_name: CLIP model name
        pretrained: Pretrained weights source
        embedding_dim: Output embedding dimension
        device: Device to use
        use_dummy: Use dummy encoder for testing
        cache_dir: Directory to cache models
        
    Returns:
        Vision encoder module
    """
    if use_dummy:
        return DummyVisionEncoder(embedding_dim=embedding_dim, device=device)
    else:
        return VisionEncoder(
            model_name=model_name,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            device=device,
            cache_dir=cache_dir,
        )


def main():
    """Test the vision encoder."""
    print("Testing VisionEncoder...")
    
    # Test with dummy encoder first
    dummy = create_vision_encoder(use_dummy=True)
    
    # Create a random image
    random_image = torch.randn(1, 3, 224, 224)
    embedding = dummy(random_image)
    
    print(f"Input shape: {random_image.shape}")
    print(f"Output shape: {embedding.shape}")
    print(f"Embedding norm: {embedding.norm(dim=-1).item():.4f}")
    print(f"Is frozen: {dummy.verify_frozen()}")
    
    # Test with real CLIP if available
    if OPEN_CLIP_AVAILABLE:
        print("\nTesting with real CLIP...")
        try:
            encoder = create_vision_encoder(
                model_name="ViT-B-32",
                pretrained="openai",
                device="cpu",
            )
            
            # Create a test image
            from PIL import Image
            test_img = Image.new("RGB", (224, 224), color="red")
            embedding = encoder.encode_pil_images([test_img])
            
            print(f"Real CLIP output shape: {embedding.shape}")
            print(f"Is frozen: {encoder.verify_frozen()}")
        except Exception as e:
            print(f"Could not load real CLIP: {e}")


if __name__ == "__main__":
    main()
