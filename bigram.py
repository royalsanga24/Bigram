import numpy as np
from collections import Counter

class BigramModel:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.bigram_counts = None
        self.char_counts = None
        self.vocab_size = 0
        
    def train(self, names):
        """Train the bigram model on a list of names."""
        # Extract all unique characters and create mappings
        all_chars = set()
        for name in names:
            for char in name.lower():
                all_chars.add(char)
        
        # Add special start/end token
        all_chars.add('.')
        
        # Create character mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.idx_to_char = {idx: char for idx, char in enumerate(sorted(all_chars))}
        self.vocab_size = len(all_chars)
        
        # Initialize count matrices
        self.bigram_counts = np.zeros((self.vocab_size, self.vocab_size))
        
        # Count bigram occurrences
        for name in names:
            name = name.lower().strip()
            # Add start and end tokens
            chars = ['.'] + list(name) + ['.']
            
            for i in range(len(chars) - 1):
                idx1 = self.char_to_idx[chars[i]]
                idx2 = self.char_to_idx[chars[i+1]]
                self.bigram_counts[idx1, idx2] += 1
        
        # Convert counts to probabilities
        self.bigram_probs = self.bigram_counts.copy()
        # Add smoothing to avoid zero probabilities
        self.bigram_probs += 1  # Add-1 smoothing
        
        # Normalize to get proper probabilities
        row_sums = self.bigram_probs.sum(axis=1, keepdims=True)
        self.bigram_probs = self.bigram_probs / row_sums
    
    def generate_name(self, max_length=10):
        """Generate a name using the trained bigram model."""
        if self.bigram_probs is None:
            return "Model not trained yet"
        
        name = []
        current_char = '.'  # Start token
        
        while True:
            current_idx = self.char_to_idx[current_char]
            probs = self.bigram_probs[current_idx]
            
            # Sample next character based on probabilities
            next_idx = np.random.choice(self.vocab_size, p=probs)
            next_char = self.idx_to_char[next_idx]
            
            # If we hit the end token or exceed max length, stop
            if next_char == '.' or len(name) >= max_length:
                break
                
            name.append(next_char)
            current_char = next_char
            
        return ''.join(name)
    
    def generate_names(self, count=10, max_length=10):
        """Generate multiple names."""
        return [self.generate_name(max_length) for _ in range(count)] 