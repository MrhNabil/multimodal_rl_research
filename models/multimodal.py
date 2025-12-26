"""
Multimodal VQA Model

Combines frozen CLIP vision encoder, trainable projection layer,
and T5-small text reasoner for visual question answering.

This is the main composed model that performs:
Image + Question → Answer
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from .vision import VisionEncoder, DummyVisionEncoder, create_vision_encoder
from .reasoning import TextReasoner, DummyTextReasoner, create_text_reasoner, GenerationOutput
from .projection import ProjectionLayer, create_projection_layer


@dataclass
class VQAOutput:
    """Output from the VQA model."""
    answers: List[str]                          # Generated answers
    token_ids: Optional[torch.Tensor] = None    # Token IDs
    log_probs: Optional[torch.Tensor] = None    # Log probabilities (for RL)
    entropy: Optional[torch.Tensor] = None      # Entropy (for RL)
    image_embeddings: Optional[torch.Tensor] = None  # CLIP embeddings


class MultimodalVQA(nn.Module):
    """
    Multimodal Visual Question Answering model.
    
    Architecture:
    1. Frozen CLIP ViT-B/32 encodes the image
    2. Trainable projection layer maps to text space
    3. T5-small generates answer given projected embedding + question
    
    The CLIP encoder is completely frozen. Only the projection layer
    and T5 model receive gradients during training.
    """
    
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_reasoner: nn.Module,
        projection_layer: nn.Module,
        prompt_template: str = "visual context: {visual} question: {question}",
    ):
        """
        Initialize the multimodal model.
        
        Args:
            vision_encoder: Frozen CLIP vision encoder
            text_reasoner: T5 text generation model
            projection_layer: Projection from vision to text space
            prompt_template: Template for constructing text prompts
        """
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_reasoner = text_reasoner
        self.projection_layer = projection_layer
        self.prompt_template = prompt_template
        
        # Ensure vision encoder is frozen
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def _construct_prompts(
        self,
        image_embeddings: torch.Tensor,
        questions: List[str],
    ) -> List[str]:
        """
        Construct text prompts from image embeddings and questions.
        
        Args:
            image_embeddings: Projected image embeddings [B, D]
            questions: List of question strings
            
        Returns:
            List of prompt strings
        """
        prompts = []
        for i, question in enumerate(questions):
            # Convert embedding to a text representation
            # (We'll use special tokens to represent the visual context)
            visual_repr = "[VISUAL]"
            prompt = self.prompt_template.format(
                visual=visual_repr,
                question=question,
            )
            prompts.append(prompt)
        return prompts
    
    def forward(
        self,
        images: torch.Tensor,
        questions: List[str],
        mode: str = "greedy",
        temperature: float = 1.0,
    ) -> VQAOutput:
        """
        Forward pass for VQA.
        
        Args:
            images: Batch of images [B, 3, H, W]
            questions: List of question strings
            mode: Generation mode ("greedy" or "sample")
            temperature: Sampling temperature (for mode="sample")
            
        Returns:
            VQAOutput with generated answers
        """
        # Extract image embeddings (frozen)
        with torch.no_grad():
            image_embeddings = self.vision_encoder(images)
        
        # Project to text space
        projected_embeddings = self.projection_layer(image_embeddings)
        
        # Construct prompts
        prompts = self._construct_prompts(projected_embeddings, questions)
        
        # Tokenize prompts
        tokens = self.text_reasoner.tokenize(prompts)
        
        # Generate answers
        if mode == "greedy":
            gen_output = self.text_reasoner.generate_greedy(
                tokens["input_ids"],
                tokens["attention_mask"],
            )
        else:  # sample
            gen_output = self.text_reasoner.generate_sample(
                tokens["input_ids"],
                tokens["attention_mask"],
                temperature=temperature,
            )
        
        return VQAOutput(
            answers=gen_output.texts,
            token_ids=gen_output.token_ids,
            log_probs=gen_output.log_probs,
            entropy=gen_output.entropy,
            image_embeddings=image_embeddings,
        )
    
    def generate(
        self,
        images: torch.Tensor,
        questions: List[str],
        temperature: float = 1.0,
        return_log_probs: bool = False,
    ) -> VQAOutput:
        """
        Generate answers for VQA (convenience method for RL training).
        
        Args:
            images: Batch of images [B, 3, H, W]
            questions: List of question strings
            temperature: Sampling temperature
            return_log_probs: Whether to return log probabilities
            
        Returns:
            VQAOutput with generated answers and log probs
        """
        mode = "sample" if return_log_probs else "greedy"
        return self.forward(images, questions, mode=mode, temperature=temperature)
    
    def compute_supervised_loss(
        self,
        images: torch.Tensor,
        questions: List[str],
        answers: List[str],
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for supervised training.
        
        Args:
            images: Batch of images [B, 3, H, W]
            questions: List of question strings
            answers: List of target answer strings
            
        Returns:
            Cross-entropy loss
        """
        # Extract image embeddings (frozen)
        with torch.no_grad():
            image_embeddings = self.vision_encoder(images)
        
        # Project to text space
        projected_embeddings = self.projection_layer(image_embeddings)
        
        # Construct prompts
        prompts = self._construct_prompts(projected_embeddings, questions)
        
        # Tokenize prompts and answers
        input_tokens = self.text_reasoner.tokenize(prompts)
        answer_tokens = self.text_reasoner.tokenize(answers)
        
        # Forward pass with labels
        outputs = self.text_reasoner(
            input_ids=input_tokens["input_ids"],
            attention_mask=input_tokens["attention_mask"],
            labels=answer_tokens["input_ids"],
        )
        
        return outputs["loss"]
    
    def get_trainable_parameters(self):
        """Get parameters that should be trained."""
        params = []
        
        # Projection layer parameters
        params.extend(self.projection_layer.parameters())
        
        # T5 parameters
        params.extend(self.text_reasoner.parameters())
        
        return params
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        projection_params = sum(p.numel() for p in self.projection_layer.parameters())
        reasoning_params = sum(p.numel() for p in self.text_reasoner.parameters())
        
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        
        return {
            "vision_encoder": vision_params,
            "projection_layer": projection_params,
            "text_reasoner": reasoning_params,
            "total": vision_params + projection_params + reasoning_params,
            "trainable": trainable_params,
        }
    
    def save(self, path: str) -> None:
        """Save trainable components."""
        torch.save({
            "projection_layer": self.projection_layer.state_dict(),
            "text_reasoner": self.text_reasoner.model.state_dict() 
                if hasattr(self.text_reasoner, 'model') else None,
        }, path)
    
    def load(self, path: str) -> None:
        """Load trainable components."""
        checkpoint = torch.load(path, map_location="cpu")
        self.projection_layer.load_state_dict(checkpoint["projection_layer"])
        if checkpoint["text_reasoner"] is not None and hasattr(self.text_reasoner, 'model'):
            self.text_reasoner.model.load_state_dict(checkpoint["text_reasoner"])


def create_multimodal_vqa(
    vision_model: str = "ViT-B-32",
    vision_pretrained: str = "openai",
    reasoning_model: str = "t5-small",
    projection_hidden: bool = False,
    projection_hidden_dim: int = 256,
    max_length: int = 32,
    device: str = "cpu",
    use_dummy: bool = False,
    cache_dir: Optional[str] = None,
) -> MultimodalVQA:
    """
    Factory function to create a MultimodalVQA model.
    
    Args:
        vision_model: CLIP model name
        vision_pretrained: CLIP pretrained weights
        reasoning_model: T5 model name
        projection_hidden: Use hidden layer in projection
        projection_hidden_dim: Hidden layer dimension
        max_length: Maximum answer length
        device: Device to use
        use_dummy: Use dummy models for testing
        cache_dir: Model cache directory
        
    Returns:
        MultimodalVQA model
    """
    # Create vision encoder
    vision_encoder = create_vision_encoder(
        model_name=vision_model,
        pretrained=vision_pretrained,
        device=device,
        use_dummy=use_dummy,
        cache_dir=cache_dir,
    )
    
    # Create text reasoner
    text_reasoner = create_text_reasoner(
        model_name=reasoning_model,
        max_length=max_length,
        device=device,
        use_dummy=use_dummy,
        cache_dir=cache_dir,
    )
    
    # Create projection layer
    vision_dim = vision_encoder.get_embedding_dim()
    text_dim = text_reasoner.get_embedding_dim()
    
    projection_layer = create_projection_layer(
        input_dim=vision_dim,
        output_dim=text_dim,
        hidden_dim=projection_hidden_dim if projection_hidden else None,
        use_hidden=projection_hidden,
    )
    
    # Create multimodal model
    model = MultimodalVQA(
        vision_encoder=vision_encoder,
        text_reasoner=text_reasoner,
        projection_layer=projection_layer,
    )
    
    return model


def main():
    """Test the multimodal VQA model."""
    print("Testing MultimodalVQA...")
    
    # Create model with dummy components
    model = create_multimodal_vqa(
        use_dummy=True,
        device="cpu",
    )
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"Parameter counts: {param_counts}")
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    questions = ["What color is the cube?", "How many spheres are there?"]
    
    # Greedy generation
    output = model(images, questions, mode="greedy")
    print(f"Greedy answers: {output.answers}")
    
    # Sampling generation
    output = model(images, questions, mode="sample", temperature=0.8)
    print(f"Sampled answers: {output.answers}")
    if output.log_probs is not None:
        print(f"Log probs shape: {output.log_probs.shape}")
    
    # Test supervised loss
    answers = ["red", "3"]
    try:
        loss = model.compute_supervised_loss(images, questions, answers)
        print(f"Supervised loss: {loss.item()}")
    except Exception as e:
        print(f"Supervised loss not available: {e}")


if __name__ == "__main__":
    main()
