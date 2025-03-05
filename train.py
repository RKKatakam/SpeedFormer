import torch
import torch.nn as nn
import tiktoken
import torch._dynamo
from transformer import Decoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = self.tokenizer.encode(text)
        
    def __len__(self):
        return len(self.tokens) - self.seq_len
        
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.decoder = Decoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            max_len=max_len
        )
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, kv_caches=None, past_length=0):
        x = self.token_embedding(x)
        x = x + self.positional_embedding[:, past_length:past_length + x.size(1)]
        x, new_kv_caches = self.decoder(x, kv_caches, past_length)
        x = self.linear(x)
        return x, new_kv_caches

def train_model(model, dataloader, criterion, optimizer, epochs, device, vocab_size):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': avg_loss})
            
        if epoch == epochs - 1:
            return avg_loss

def main():
    # Hyperparameters
    d_model = 256
    n_heads = 4
    n_layers = 2
    max_len = 128
    batch_size = 128
    learning_rate = 1e-3
    epochs = 7
    compile_model = True  
    
    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    
    # Load sample text (replace with your dataset)
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset and dataloader
    dataset = TextDataset(text, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(vocab_size, d_model, n_heads, n_layers, max_len).to(device)

    
    # Try to compile model if enabled
    if compile_model:
        #model = torch.compile(model)
        print("Model compiled")
            
 
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train model
    final_loss = train_model(model, dataloader, criterion, optimizer, epochs, device, vocab_size)
    print(f"Final loss: {final_loss}")
    
    # Save model
    torch.save(model.state_dict(), 'speedformer_model.pt')
    
    # Generate sample text
    model.eval()
    prompt = "Harry don't "
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    max_new_tokens = 100
    generated_tokens = []
    kv_caches = None
    past_length = 0
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, kv_caches = model(input_ids[:, -1:], kv_caches, past_length)
            next_token = torch.argmax(logits[:, -1], dim=-1)
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            past_length += 1
    
    generated_text = tokenizer.decode(generated_tokens)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()
