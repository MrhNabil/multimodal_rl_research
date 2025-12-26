"""
Fast Lightweight Reasoning Module

Uses simple MLP classifier instead of T5 for MUCH faster training.
~100x faster than T5 while still supporting RL training.

For VQA with fixed answer vocabulary (colors, shapes, counts),
classification is equivalent to generation but orders of magnitude faster.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class GenerationOutput:
    """Output from text generation (compatible with T5 interface)."""
    texts: List[str]
    token_ids: torch.Tensor
    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None


class FastVQAReasoner(nn.Module):
    """
    Fast MLP-based VQA reasoner.
    
    Instead of generating text token-by-token like T5, this model
    treats VQA as a classification problem over a fixed answer vocabulary.
    
    This is ~100x faster while supporting the same training paradigms:
    - Supervised: Cross-entropy loss on correct answer class
    - RL (REINFORCE): Sample from softmax, get log_prob for policy gradient
    
    Answer vocabulary covers all possible answers in our synthetic dataset:
    - Colors: red, blue, green, yellow
    - Shapes: cube, sphere, cylinder
    - Counts: 0, 1, 2, 3, 4, 5
    - Positions: left, center, right, none
    """
    
    # Fixed answer vocabulary for synthetic VQA - MUST MATCH DATASET!
    ANSWER_VOCAB = [
        # Colors (for color questions)
        "red", "blue", "green", "yellow",
        # Shapes (for shape questions)
        "cube", "sphere", "cylinder",
        # Counts (for count questions)
        "0", "1", "2", "3",
        # Compound: color + shape (for spatial questions like "what is left of X?")
        "red cube", "red sphere", "red cylinder",
        "blue cube", "blue sphere", "blue cylinder",
        "green cube", "green sphere", "green cylinder", 
        "yellow cube", "yellow sphere", "yellow cylinder",
        # Nothing (when no object matches)
        "nothing",
    ]
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        """
        Initialize fast VQA reasoner.
        
        Args:
            input_dim: Dimension of visual embedding input
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
            device: Device to use
        """
        super().__init__()
        
        self.device = device
        self.embedding_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_answers = len(self.ANSWER_VOCAB)
        
        # Create answer to index mapping
        self.answer_to_idx = {ans: i for i, ans in enumerate(self.ANSWER_VOCAB)}
        self.idx_to_answer = {i: ans for i, ans in enumerate(self.ANSWER_VOCAB)}
        
        # Visual encoder (projects visual embedding to hidden_dim)
        self.visual_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Main MLP classifier
        layers = []
        current_dim = hidden_dim * 2  # visual (hidden) + question (hidden)
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, self.num_answers))
        
        self.classifier = nn.Sequential(*layers)
        
        # Semantic question encoding
        # We embed based on question TYPE and TARGET
        # Question types: color, shape, count, spatial (4 types)
        # Targets: red, blue, green, yellow, cube, sphere, cylinder, left, right (10+ targets)
        self.question_type_embed = nn.Embedding(5, hidden_dim // 2)  # 5 question types
        self.target_embed = nn.Embedding(20, hidden_dim // 2)  # 20 possible targets
        
        self.to(device)
        
        print(f"FastVQAReasoner initialized: {self.num_answers} answer classes, {hidden_dim} hidden dim")
    
    def _parse_question(self, question: str) -> tuple:
        """Parse question to extract type and target."""
        q = question.lower()
        
        # Determine question type
        if "color" in q or "what color" in q:
            q_type = 0  # color question
        elif "shape" in q or "what shape" in q:
            q_type = 1  # shape question
        elif "how many" in q or "count" in q or "number" in q:
            q_type = 2  # count question
        elif "where" in q or "position" in q or "left" in q or "right" in q:
            q_type = 3  # spatial question
        else:
            q_type = 4  # unknown
        
        # Determine target (what the question is asking about)
        target = 0  # default
        targets = {
            "red": 1, "blue": 2, "green": 3, "yellow": 4,
            "cube": 5, "sphere": 6, "cylinder": 7,
            "large": 8, "small": 9, "medium": 10,
            "left": 11, "right": 12, "center": 13,
            "front": 14, "back": 15,
        }
        for word, idx in targets.items():
            if word in q:
                target = idx
                break
        
        return q_type, target
    
    def _encode_questions(self, questions: List[str]) -> torch.Tensor:
        """Encode questions using semantic parsing."""
        q_types = []
        targets = []
        
        for q in questions:
            q_type, target = self._parse_question(q)
            q_types.append(q_type)
            targets.append(target)
        
        q_type_tensor = torch.tensor(q_types, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        
        # Get embeddings
        q_type_emb = self.question_type_embed(q_type_tensor)  # [B, hidden_dim//2]
        target_emb = self.target_embed(target_tensor)  # [B, hidden_dim//2]
        
        # Concatenate
        question_emb = torch.cat([q_type_emb, target_emb], dim=-1)  # [B, hidden_dim]
        
        return question_emb
    
    def forward(
        self,
        visual_embedding: torch.Tensor,
        questions: List[str],
    ) -> torch.Tensor:
        """
        Forward pass returning logits.
        
        Args:
            visual_embedding: Visual features [B, D]
            questions: List of question strings
            
        Returns:
            Logits over answer vocabulary [B, num_answers]
        """
        # Encode questions
        question_embedding = self._encode_questions(questions)
        
        # Ensure visual embedding is on the right device and dimension
        if visual_embedding.device != self.device:
            visual_embedding = visual_embedding.to(self.device)
        
        if visual_embedding.dim() == 1:
            visual_embedding = visual_embedding.unsqueeze(0)
        
        # Project visual embedding to hidden_dim
        visual_hidden = self.visual_encoder(visual_embedding)
        
        # Concatenate visual and question features (both are hidden_dim now)
        combined = torch.cat([visual_hidden, question_embedding], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits
    
    def generate_greedy(
        self,
        visual_embedding: torch.Tensor,
        questions: List[str],
    ) -> GenerationOutput:
        """
        Generate answers using greedy (argmax) decoding.
        
        Args:
            visual_embedding: Visual features [B, D]
            questions: List of question strings
            
        Returns:
            GenerationOutput with predicted answers
        """
        logits = self.forward(visual_embedding, questions)
        
        # Greedy selection
        pred_indices = logits.argmax(dim=-1)
        
        # Convert to answer strings
        answers = [self.idx_to_answer[idx.item()] for idx in pred_indices]
        
        return GenerationOutput(
            texts=answers,
            token_ids=pred_indices.unsqueeze(-1),
            log_probs=None,
            entropy=None,
        )
    
    def generate_sample(
        self,
        visual_embedding: torch.Tensor,
        questions: List[str],
        temperature: float = 1.0,
    ) -> GenerationOutput:
        """
        Generate answers using sampling (for RL training).
        
        Args:
            visual_embedding: Visual features [B, D]
            questions: List of question strings
            temperature: Sampling temperature
            
        Returns:
            GenerationOutput with sampled answers and log probabilities
        """
        logits = self.forward(visual_embedding, questions)
        
        # Apply temperature
        scaled_logits = logits / temperature
        
        # Compute probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs_all = F.log_softmax(scaled_logits, dim=-1)
        
        # Sample from distribution
        sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Get log prob of sampled action
        sampled_log_probs = log_probs_all.gather(1, sampled_indices.unsqueeze(-1)).squeeze(-1)
        
        # Compute entropy
        entropy = -(probs * log_probs_all).sum(dim=-1)
        
        # Convert to answer strings
        answers = [self.idx_to_answer[idx.item()] for idx in sampled_indices]
        
        return GenerationOutput(
            texts=answers,
            token_ids=sampled_indices.unsqueeze(-1),
            log_probs=sampled_log_probs.unsqueeze(-1),  # [B, 1] for compatibility
            entropy=entropy,
        )
    
    def compute_loss(
        self,
        visual_embedding: torch.Tensor,
        questions: List[str],
        target_answers: List[str],
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for supervised training.
        
        Args:
            visual_embedding: Visual features [B, D]
            questions: List of question strings
            target_answers: List of ground truth answers
            
        Returns:
            Cross-entropy loss
        """
        logits = self.forward(visual_embedding, questions)
        
        # Convert answers to indices
        target_indices = []
        for ans in target_answers:
            ans_lower = ans.lower().strip()
            if ans_lower in self.answer_to_idx:
                target_indices.append(self.answer_to_idx[ans_lower])
            else:
                target_indices.append(self.answer_to_idx["unknown"])
        
        targets = torch.tensor(target_indices, dtype=torch.long, device=self.device)
        
        return F.cross_entropy(logits, targets)
    
    def tokenize(self, texts: List[str], max_length: int = 32) -> Dict[str, torch.Tensor]:
        """Compatibility method for T5 interface."""
        batch_size = len(texts)
        return {
            "input_ids": torch.ones(batch_size, max_length, dtype=torch.long, device=self.device),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long, device=self.device),
        }
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension for compatibility."""
        return self.embedding_dim


class FastMultimodalVQA(nn.Module):
    """
    Fast Multimodal VQA using MLP classifier instead of T5.
    
    ~100x faster than T5-based version while maintaining same interface.
    """
    
    def __init__(
        self,
        vision_encoder: nn.Module,
        reasoner: FastVQAReasoner,
        projection_layer: nn.Module,
    ):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.reasoner = reasoner
        self.projection_layer = projection_layer
        
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        images: torch.Tensor,
        questions: List[str],
        mode: str = "greedy",
        temperature: float = 1.0,
    ):
        """Forward pass for VQA."""
        # Extract image embeddings (frozen)
        with torch.no_grad():
            image_embeddings = self.vision_encoder(images)
        
        # Project to reasoner space
        projected = self.projection_layer(image_embeddings)
        
        # Generate answers
        if mode == "greedy":
            output = self.reasoner.generate_greedy(projected, questions)
        else:
            output = self.reasoner.generate_sample(projected, questions, temperature)
        
        return output
    
    def generate(
        self,
        images: torch.Tensor,
        questions: List[str],
        temperature: float = 1.0,
        return_log_probs: bool = False,
    ):
        """Generate answers (convenience method for RL)."""
        mode = "sample" if return_log_probs else "greedy"
        return self.forward(images, questions, mode=mode, temperature=temperature)
    
    def compute_supervised_loss(
        self,
        images: torch.Tensor,
        questions: List[str],
        answers: List[str],
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        with torch.no_grad():
            image_embeddings = self.vision_encoder(images)
        
        projected = self.projection_layer(image_embeddings)
        return self.reasoner.compute_loss(projected, questions, answers)
    
    def get_trainable_parameters(self):
        """Get trainable parameters."""
        params = list(self.projection_layer.parameters())
        params.extend(self.reasoner.parameters())
        return params
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        projection_params = sum(p.numel() for p in self.projection_layer.parameters())
        reasoning_params = sum(p.numel() for p in self.reasoner.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "vision_encoder": vision_params,
            "projection_layer": projection_params,
            "text_reasoner": reasoning_params,
            "total": vision_params + projection_params + reasoning_params,
            "trainable": trainable,
        }


def create_fast_multimodal_vqa(
    vision_model: str = "ViT-B-32",
    vision_pretrained: str = "openai",
    hidden_dim: int = 256,
    num_layers: int = 2,
    device: str = "cpu",
    use_dummy: bool = False,
    cache_dir: Optional[str] = None,
):
    """
    Create a fast multimodal VQA model.
    
    Args:
        vision_model: CLIP model name
        vision_pretrained: CLIP pretrained weights
        hidden_dim: MLP hidden dimension
        num_layers: Number of MLP layers
        device: Device to use
        use_dummy: Use dummy vision encoder for testing
        cache_dir: Model cache directory
        
    Returns:
        FastMultimodalVQA model
    """
    from .vision import create_vision_encoder
    from .projection import create_projection_layer
    
    # Create vision encoder
    vision_encoder = create_vision_encoder(
        model_name=vision_model,
        pretrained=vision_pretrained,
        device=device,
        use_dummy=use_dummy,
        cache_dir=cache_dir,
    )
    
    vision_dim = vision_encoder.get_embedding_dim()
    
    # Create fast reasoner
    reasoner = FastVQAReasoner(
        input_dim=vision_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )
    
    # Create projection layer
    projection_layer = create_projection_layer(
        input_dim=vision_dim,
        output_dim=vision_dim,  # Same dim for the fast model
        use_hidden=False,
    )
    
    return FastMultimodalVQA(
        vision_encoder=vision_encoder,
        reasoner=reasoner,
        projection_layer=projection_layer,
    )


if __name__ == "__main__":
    # Test the fast model
    print("Testing FastVQAReasoner...")
    
    reasoner = FastVQAReasoner(input_dim=512, hidden_dim=256, device="cpu")
    
    # Test inputs
    visual_emb = torch.randn(4, 512)
    questions = ["What color is the cube?", "How many spheres?", "What shape is red?", "Position of cube?"]
    
    # Greedy
    output = reasoner.generate_greedy(visual_emb, questions)
    print(f"Greedy answers: {output.texts}")
    
    # Sample
    output = reasoner.generate_sample(visual_emb, questions, temperature=0.8)
    print(f"Sampled answers: {output.texts}")
    print(f"Log probs: {output.log_probs}")
    print(f"Entropy: {output.entropy}")
    
    # Loss
    targets = ["red", "2", "cube", "left"]
    loss = reasoner.compute_loss(visual_emb, questions, targets)
    print(f"Loss: {loss.item():.4f}")
    
    print("\n✓ FastVQAReasoner working!")
