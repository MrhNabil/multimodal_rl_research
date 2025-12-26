"""
Text Reasoning Module (T5-small)

Wraps the T5-small model for text generation given visual context.
Supports both greedy decoding (for evaluation) and sampling (for RL).

This represents "Atomic Skill B" in the research design.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")


@dataclass
class GenerationOutput:
    """Output from text generation."""
    texts: List[str]              # Generated text strings
    token_ids: torch.Tensor       # Token IDs [B, L]
    log_probs: Optional[torch.Tensor] = None  # Log probabilities [B, L]
    entropy: Optional[torch.Tensor] = None    # Entropy [B]


class TextReasoner(nn.Module):
    """
    T5-small based text reasoner.
    
    Takes a text prompt (constructed from visual embedding + question)
    and generates an answer. Supports both deterministic and stochastic
    generation for supervised and RL training respectively.
    """
    
    def __init__(
        self,
        model_name: str = "t5-small",
        max_length: int = 32,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the text reasoner.
        
        Args:
            model_name: HuggingFace model name (t5-small, t5-base, etc.)
            max_length: Maximum generation length
            device: Device to use ("cpu" for this project)
            cache_dir: Directory to cache pretrained models
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")
        
        # Load tokenizer and model
        print(f"Loading T5 model: {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            legacy=False,
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model.to(device)
        
        # Get model dimensions
        self.embedding_dim = self.model.config.d_model
        self.vocab_size = self.model.config.vocab_size
        
        print(f"T5 model loaded. Embedding dim: {self.embedding_dim}, Vocab size: {self.vocab_size}")
    
    def tokenize(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: List of input strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        max_length = max_length or self.max_length
        
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }
    
    def get_encoder_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get T5 encoder outputs.
        
        Args:
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            
        Returns:
            Encoder hidden states [B, L, D]
        """
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return encoder_outputs.last_hidden_state
    
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> GenerationOutput:
        """
        Generate text using greedy decoding.
        
        Args:
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            max_length: Maximum generation length
            
        Returns:
            GenerationOutput with generated texts
        """
        max_length = max_length or self.max_length
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=False,
                num_beams=1,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return GenerationOutput(
            texts=texts,
            token_ids=outputs,
            log_probs=None,
            entropy=None,
        )
    
    def generate_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
        max_length: Optional[int] = None,
    ) -> GenerationOutput:
        """
        Generate text using sampling (for RL training).
        
        Args:
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            temperature: Sampling temperature
            max_length: Maximum generation length
            
        Returns:
            GenerationOutput with generated texts and log probabilities
        """
        max_length = max_length or self.max_length
        batch_size = input_ids.shape[0]
        
        # Get encoder outputs
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Initialize decoder input
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )
        
        # Storage for outputs
        all_token_ids = []
        all_log_probs = []
        all_entropies = []
        
        # Autoregressive generation
        for step in range(max_length):
            # Get logits
            outputs = self.model(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Compute log probabilities
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Compute entropy
            entropy = -(probs * log_probs).sum(dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Get log prob of sampled token
            sampled_log_prob = log_probs.gather(1, next_token).squeeze(-1)
            
            # Store
            all_token_ids.append(next_token)
            all_log_probs.append(sampled_log_prob)
            all_entropies.append(entropy)
            
            # Update decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Check for EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        # Stack outputs
        token_ids = torch.cat(all_token_ids, dim=1)
        log_probs = torch.stack(all_log_probs, dim=1)
        entropies = torch.stack(all_entropies, dim=1).mean(dim=1)
        
        # Decode texts
        texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        
        return GenerationOutput(
            texts=texts,
            token_ids=token_ids,
            log_probs=log_probs,
            entropy=entropies,
        )
    
    def compute_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability of target tokens given input.
        
        Args:
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            target_ids: Target token IDs [B, T]
            
        Returns:
            Log probabilities [B]
        """
        # Create decoder input (shift right)
        decoder_input_ids = self.model._shift_right(target_ids)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        
        # Compute log probs
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        
        # Get log prob of each target token
        target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        
        # Mask padding
        target_mask = (target_ids != self.tokenizer.pad_token_id).float()
        
        # Sum log probs (masked)
        sequence_log_prob = (target_log_probs * target_mask).sum(dim=1)
        
        return sequence_log_prob
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for supervised training.
        
        Args:
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            labels: Target token IDs [B, T] (optional)
            
        Returns:
            Dictionary with loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }
    
    def get_embedding_dim(self) -> int:
        """Return the model's embedding dimension."""
        return self.embedding_dim


class DummyTextReasoner(nn.Module):
    """
    Dummy text reasoner for testing without T5.
    
    Returns random outputs but supports gradient flow for training tests.
    Useful for testing the pipeline without loading the full T5 model.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        vocab_size: int = 32128,
        device: str = "cpu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.device = device
        
        # Add trainable parameters so gradients can flow
        self.dummy_linear = nn.Linear(embedding_dim, vocab_size)
        self.dummy_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        print("Using DummyTextReasoner (random outputs with gradient support)")
    
    def tokenize(self, texts: List[str], max_length: int = 32) -> Dict[str, torch.Tensor]:
        """Return dummy tokens."""
        batch_size = len(texts)
        return {
            "input_ids": torch.ones(batch_size, max_length, dtype=torch.long, device=self.device),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long, device=self.device),
        }
    
    def generate_greedy(self, input_ids, attention_mask, max_length=32) -> GenerationOutput:
        """Return dummy generation."""
        batch_size = input_ids.shape[0]
        return GenerationOutput(
            texts=["dummy answer"] * batch_size,
            token_ids=torch.ones(batch_size, 3, dtype=torch.long, device=self.device),
        )
    
    def generate_sample(self, input_ids, attention_mask, temperature=1.0, max_length=32) -> GenerationOutput:
        """Return dummy generation with gradient-supporting log probs."""
        batch_size = input_ids.shape[0]
        seq_len = 3
        
        # Create embeddings that go through trainable layer for gradient flow
        dummy_input = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        logits = self.dummy_linear(dummy_input)  # [B, vocab_size]
        
        # Compute log probs (will have grad_fn due to dummy_linear)
        log_probs_vocab = torch.log_softmax(logits / temperature, dim=-1)
        
        # Take first 3 tokens' log probs and expand to sequence
        log_probs = log_probs_vocab[:, :seq_len]  # [B, seq_len]
        
        # Entropy computation
        probs = torch.softmax(logits / temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [B]
        
        return GenerationOutput(
            texts=["dummy answer"] * batch_size,
            token_ids=torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device),
            log_probs=log_probs,
            entropy=entropy,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for supervised training."""
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Create dummy logits with gradient support
        dummy_input = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        logits = self.dummy_linear(dummy_input)  # [B, vocab_size]
        logits = logits.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, vocab_size]
        
        loss = None
        if labels is not None:
            # Compute cross-entropy loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


def create_text_reasoner(
    model_name: str = "t5-small",
    max_length: int = 32,
    device: str = "cpu",
    use_dummy: bool = False,
    cache_dir: Optional[str] = None,
) -> nn.Module:
    """
    Factory function to create a text reasoner.
    
    Args:
        model_name: HuggingFace model name
        max_length: Maximum generation length
        device: Device to use
        use_dummy: Use dummy reasoner for testing
        cache_dir: Directory to cache models
        
    Returns:
        Text reasoner module
    """
    if use_dummy:
        return DummyTextReasoner(device=device)
    else:
        return TextReasoner(
            model_name=model_name,
            max_length=max_length,
            device=device,
            cache_dir=cache_dir,
        )


def main():
    """Test the text reasoner."""
    print("Testing TextReasoner...")
    
    # Test with dummy first
    dummy = create_text_reasoner(use_dummy=True)
    tokens = dummy.tokenize(["What color is the cube?"])
    output = dummy.generate_greedy(tokens["input_ids"], tokens["attention_mask"])
    print(f"Dummy output: {output.texts}")
    
    # Test with real T5 if available
    if TRANSFORMERS_AVAILABLE:
        print("\nTesting with real T5...")
        try:
            reasoner = create_text_reasoner(
                model_name="t5-small",
                device="cpu",
            )
            
            # Test generation
            tokens = reasoner.tokenize(["question: What color is the sky? context: The sky is blue."])
            
            # Greedy generation
            greedy_output = reasoner.generate_greedy(
                tokens["input_ids"],
                tokens["attention_mask"],
            )
            print(f"Greedy output: {greedy_output.texts}")
            
            # Sampling
            sample_output = reasoner.generate_sample(
                tokens["input_ids"],
                tokens["attention_mask"],
                temperature=0.8,
            )
            print(f"Sample output: {sample_output.texts}")
            print(f"Log probs shape: {sample_output.log_probs.shape}")
            print(f"Entropy: {sample_output.entropy}")
            
        except Exception as e:
            print(f"Could not load real T5: {e}")


if __name__ == "__main__":
    main()
