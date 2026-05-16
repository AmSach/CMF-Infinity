import re
from collections import Counter
from typing import Dict, List, Set, Tuple

import torch

class SimpleBPETokenizer:
    """A simple, self-contained BPE tokenizer for the CMF project."""
    
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.byte_encoder = {i: chr(i) for i in range(256)}
        self.byte_decoder = {chr(i): i for i in range(256)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[bytes, bytes], int] = {}
        self.token_to_id: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}

    def train(self, text: str):
        print(f"Training BPE tokenizer on {len(text)} characters...")
        # Start with bytes
        tokens = list(text.encode("utf-8"))
        
        current_vocab_size = 256
        while current_vocab_size < self.vocab_size:
            # Count pairs
            pairs = Counter()
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i+1])] += 1
            
            if not pairs:
                break
                
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break
                
            # Merge
            new_token_id = current_vocab_size
            new_token_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            self.vocab[new_token_id] = new_token_bytes
            self.merges[best_pair] = new_token_id
            self.token_to_id[new_token_bytes] = new_token_id
            
            # Update tokens list
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            current_vocab_size += 1
            
        print(f"BPE Training complete. Final vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> torch.Tensor:
        tokens = list(text.encode("utf-8"))
        
        # Apply merges in the order they were created
        for pair, new_id in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.detach().cpu().flatten().tolist()
        res = b""
        for idx in ids:
            if idx in self.vocab:
                res += self.vocab[idx]
            else:
                res += b"?"
        return res.decode("utf-8", errors="replace")
