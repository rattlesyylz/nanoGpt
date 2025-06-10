"""
Inference script for GPT model
Loads a trained model and generates text
"""
import os
import json
import pickle
import torch
from model import GPT, GPTConfig

__all__ = ['TextGenerator']

class TextGenerator:
    def __init__(self, model_path, config_path=None, tokenizer_path=None):
        """
        Initialize the text generator
        
        Args:
            model_path: Path to the saved model checkpoint
            config_path: Path to config.json (optional, will try to find it)
            tokenizer_path: Path to tokenizer.pkl (optional, will try to find it)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        print(f"Loading model from {model_path}")
        try:
            # Try with weights_only=False for PyTorch 2.6+ compatibility
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't have weights_only parameter
            checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load model configuration
        if config_path is None:
            # Try to find config in the same directory as model
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, 'config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            print("Config file not found, using config from checkpoint")
            self.config = checkpoint.get('config', {})
        
        # Load tokenizer
        if tokenizer_path is None:
            # Try to find tokenizer in the same directory as model
            model_dir = os.path.dirname(model_path)
            tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
        
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                tokenizer_data = pickle.load(f)
                self.char_to_idx = tokenizer_data['char_to_idx']
                self.idx_to_char = tokenizer_data['idx_to_char']
                self.chars = tokenizer_data['chars']
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
        
        # Initialize model
        model_args = checkpoint['model_args']
        self.model = GPT(model_args)
        
        # Load model weights
        state_dict = checkpoint['model']
        # Remove any unexpected keys
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        self.model.load_state_dict(filtered_state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {len(self.chars)}")
        print(f"Model parameters: {self.model.get_num_params()/1e6:.2f}M")
    
    def encode(self, text):
        """Encode text to token indices"""
        if any(ch not in self.char_to_idx for ch in text):
            raise ValueError(f"Unknown character(s) in input: {set(ch for ch in text if ch not in self.char_to_idx)}")
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        """Decode token indices to text"""
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])
    
    def generate(self, prompt="", max_new_tokens=100, temperature=1.0, top_k=None):
        """
        Generate text given a prompt
        
        Args:
            prompt: Input text to start generation from
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
        
        Returns:
            Generated text string
        """
        # Encode the prompt
        if prompt:
            context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            # Start with empty context
            context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        
        print(f"Generating with prompt: '{prompt}'")
        print(f"Parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}")
        print("-" * 50)
        
        # Generate tokens
        generated = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if it's too long
                context_cropped = context if context.size(1) <= self.model.config.block_size else context[:, -self.model.config.block_size:]
                
                # Forward pass
                logits = self.model(context_cropped)
                logits = logits[:, -1, :] / temperature  # Take last token and scale by temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample from the distribution
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to context and generated sequence
                context = torch.cat((context, next_token), dim=1)
                generated.append(next_token.item())
        
        # Decode the generated tokens
        generated_text = self.decode(generated)
        full_text = prompt + generated_text
        
        return full_text, generated_text
    
    def complete_text(self, prompt, max_new_tokens=50, temperature=0.8):
        """
        Complete a given text prompt
        """
        full_text, generated = self.generate(prompt, max_new_tokens, temperature)
        
        print(f"Input: {prompt}")
        print(f"Generated: {generated}")
        print(f"Full text: {full_text}")
        
        return full_text
    
    def sample_unconditional(self, max_tokens=100, temperature=1.0):
        """
        Generate text unconditionally (no prompt)
        """
        full_text, generated = self.generate("", max_tokens, temperature)
        
        print(f"Unconditional sample: {full_text}")
        
        return full_text


def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate text using trained GPT model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--config_path', type=str, default=None, help='Path to config file')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to tokenizer file')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TextGenerator(
        model_path=args.model_path,
        config_path=args.config_path,
        tokenizer_path=args.tokenizer_path
    )
    
    # Generate text
    if args.prompt:
        full_text = generator.complete_text(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
    else:
        full_text = generator.sample_unconditional(
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
    
    return full_text


def test_sanity_check(model_path, temperature=0.8):
    """
    Test function for sanity check - should reproduce the memorized string
    """
    print("=" * 60)
    print("SANITY CHECK: Testing if model can reproduce memorized text")
    print("=" * 60)
    
    generator = TextGenerator(model_path)
    
    # Test with empty prompt (unconditional generation)
    print("\n1. Unconditional generation:")
    result = generator.sample_unconditional(max_tokens=50, temperature=temperature)  # Low temperature for deterministic output
    
    # Test with partial prompt
    print("\n2. Completing partial prompt:")
    partial_prompts = ["I", "I love", "I love machine"]
    for prompt in partial_prompts:
        print(f"\nPrompt: '{prompt}'")
        result = generator.complete_text(prompt, max_new_tokens=20, temperature=temperature)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["test", "generate"])
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--prompt", default="", help="Prompt for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    if args.mode == "test":
        test_sanity_check(args.model_path, temperature=args.temperature)
    else:
        generator = TextGenerator(args.model_path)
        result = generator.complete_text(args.prompt, temperature=args.temperature)
        print(result)
