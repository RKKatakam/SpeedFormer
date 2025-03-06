import torch
import torch.nn.functional as F 
import torch.nn as nn
import tiktoken
import torch._dynamo
from transformer import Decoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

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
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.decoder = Decoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=4 * d_model,  # Added explicit d_ff parameter
            max_len=max_len,
            dropout=0.1  # Added explicit dropout parameter
        )
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, kv_caches=None, past_length=0):
        x = self.token_embedding(x)
        x = x + self.positional_embedding[:, past_length:past_length + x.size(1)]
        x, new_kv_caches = self.decoder(x, kv_caches, past_length)
        x = self.linear(x)
        return x, new_kv_caches

def train_model(model, dataloader, criterion, optimizer, epochs, device, vocab_size, checkpoint_path=None):
    model.train()
    best_loss = float('inf')
    
    # Load checkpoint if exists
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"Resuming from epoch {start_epoch} with loss {best_loss}")
    
    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{start_epoch+epochs}')
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Added gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': avg_loss})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{start_epoch+epochs}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, 'speedformer_best_checkpoint.pt')
            
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }, 'speedformer_latest_checkpoint.pt')
            
    return best_loss

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, device="cuda"):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    generated_tokens = []
    kv_caches = None
    past_length = 0
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, kv_caches = model(input_ids[:, -1:], kv_caches, past_length)
            
            # Apply temperature and sample
            if temperature > 0:
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
                
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            past_length += 1
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eot_token:
                break
    
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text

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
    
    # Ensure data file exists
    data_file = 'data.txt'
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return
    
    # Load sample text
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset and dataloader
    dataset = TextDataset(text, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GPTModel(vocab_size, d_model, n_heads, n_layers, max_len).to(device)

    # Try to compile model if enabled
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model compiled successfully")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Continuing with uncompiled model")
 
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Train model
    final_loss = train_model(
        model, dataloader, criterion, optimizer, 
        epochs, device, vocab_size, 
        checkpoint_path='speedformer_latest_checkpoint.pt'
    )
    print(f"Final loss: {final_loss}")
    
    # Save final model
    torch.save(model.state_dict(), 'speedformer_model.pt')
    
    # Generate sample text
    import torch.nn.functional as F  # Added import
    prompt = "Harry don't "
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=100, device=device)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()