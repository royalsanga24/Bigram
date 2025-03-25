import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class NameDataset(Dataset):
    def __init__(self, names, char_to_idx):
        self.names = names
        self.char_to_idx = char_to_idx
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = ['.'] + list(self.names[idx].lower()) + ['.']
        x = torch.tensor([self.char_to_idx[c] for c in name[:-1]], dtype=torch.long)
        y = torch.tensor([self.char_to_idx[c] for c in name[1:]], dtype=torch.long)
        return x, y

def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    # Separate inputs and targets
    sequences, targets = zip(*batch)
    
    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return sequences_padded, targets_padded

class NameGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        output = self.fc(lstm_out)
        return output, hidden
    
class NeuralModel:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, names, epochs=50):
        # Create vocabulary
        all_chars = set()
        for name in names:
            for char in name.lower():
                all_chars.add(char)
        all_chars.add('.')
        
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(all_chars)
        
        # Create model
        self.model = NameGenerator(self.vocab_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
        
        # Create dataset
        dataset = NameDataset(names, self.char_to_idx)
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True,
            collate_fn=collate_fn  # Use custom collate function
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output, _ = self.model(x)
                loss = criterion(output.view(-1, self.vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
    def generate_name(self, max_length=10):
        if self.model is None:
            return "Model not trained yet"
        
        self.model.eval()
        with torch.no_grad():
            current_char = '.'
            name = []
            hidden = None
            
            while True:
                x = torch.tensor([[self.char_to_idx[current_char]]], device=self.device)
                output, hidden = self.model(x, hidden)
                probs = torch.softmax(output[0, -1], dim=0)
                next_idx = torch.multinomial(probs, 1).item()
                next_char = self.idx_to_char[next_idx]
                
                if next_char == '.' or len(name) >= max_length:
                    break
                    
                name.append(next_char)
                current_char = next_char
                
            return ''.join(name)
    
    def generate_names(self, count=10, max_length=10):
        return [self.generate_name(max_length) for _ in range(count)] 