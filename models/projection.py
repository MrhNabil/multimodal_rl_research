"""
Projection Layer Module

Projects CLIP image embeddings into the T5 text space.
This is the key learnable component that bridges vision and language.
"""

import torch
import torch.nn as nn
from typing import Optional


class ProjectionLayer(nn.Module):
    """
    Projects vision embeddings to text embedding space.
    
    Maps CLIP's 512-dim embeddings to T5's embedding space.
    Can use either a simple linear layer or a two-layer MLP.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: Optional[int] = None,
        use_hidden: bool = False,
        dropout: float = 0.1,
    ):
        """
        Initialize the projection layer.
        
        Args:
            input_dim: Input dimension (CLIP embedding dim)
            output_dim: Output dimension (T5 embedding dim)
            hidden_dim: Hidden layer dimension (if use_hidden=True)
            use_hidden: Whether to use a hidden layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_hidden = use_hidden
        
        if use_hidden:
            hidden_dim = hidden_dim or (input_dim + output_dim) // 2
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.projection = nn.Linear(input_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings.
        
        Args:
            x: Input embeddings [B, input_dim]
            
        Returns:
            Projected embeddings [B, output_dim]
        """
        return self.projection(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTokenProjection(nn.Module):
    """
    Projects vision embeddings to multiple tokens.
    
    Instead of a single embedding, produces N token embeddings
    that can be prepended to the text input. This gives the
    model more capacity to represent visual information.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        num_tokens: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-token projection.
        
        Args:
            input_dim: Input dimension (CLIP embedding dim)
            output_dim: Output dimension per token (T5 embedding dim)
            num_tokens: Number of tokens to produce
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        
        # Project to num_tokens * output_dim, then reshape
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * num_tokens),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Optional layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to multiple tokens.
        
        Args:
            x: Input embeddings [B, input_dim]
            
        Returns:
            Token embeddings [B, num_tokens, output_dim]
        """
        batch_size = x.shape[0]
        
        # Project
        projected = self.projection(x)
        
        # Reshape to [B, num_tokens, output_dim]
        tokens = projected.view(batch_size, self.num_tokens, self.output_dim)
        
        # Apply layer norm
        tokens = self.layer_norm(tokens)
        
        return tokens
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_projection_layer(
    input_dim: int = 512,
    output_dim: int = 512,
    hidden_dim: Optional[int] = None,
    use_hidden: bool = False,
    num_tokens: int = 1,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Factory function to create a projection layer.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden dimension (for MLP)
        use_hidden: Whether to use hidden layer
        num_tokens: Number of output tokens
        dropout: Dropout probability
        
    Returns:
        Projection layer module
    """
    if num_tokens > 1:
        return MultiTokenProjection(
            input_dim=input_dim,
            output_dim=output_dim,
            num_tokens=num_tokens,
            dropout=dropout,
        )
    else:
        return ProjectionLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            use_hidden=use_hidden,
            dropout=dropout,
        )


def main():
    """Test the projection layers."""
    print("Testing ProjectionLayer...")
    
    # Simple projection
    proj = ProjectionLayer(input_dim=512, output_dim=512)
    x = torch.randn(4, 512)
    y = proj(x)
    print(f"Simple projection: {x.shape} -> {y.shape}")
    print(f"Parameters: {proj.count_parameters()}")
    
    # MLP projection
    mlp_proj = ProjectionLayer(
        input_dim=512,
        output_dim=512,
        hidden_dim=256,
        use_hidden=True,
    )
    y = mlp_proj(x)
    print(f"MLP projection: {x.shape} -> {y.shape}")
    print(f"Parameters: {mlp_proj.count_parameters()}")
    
    # Multi-token projection
    multi_proj = MultiTokenProjection(
        input_dim=512,
        output_dim=512,
        num_tokens=4,
    )
    y = multi_proj(x)
    print(f"Multi-token projection: {x.shape} -> {y.shape}")
    print(f"Parameters: {multi_proj.count_parameters()}")


if __name__ == "__main__":
    main()
