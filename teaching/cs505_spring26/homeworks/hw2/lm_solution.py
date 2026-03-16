import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer, MultiHeadTransformer
import sys
from dataset import load_data, get_tokenizer, get_translation_data

def train(model, train_data, dev_data, epochs=1, lr=1e-3, device='cpu', save_path='model.pt'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_data):
            # batch is expected to be (B, T) tensor of token indices
            inputs = batch[:, :-1].to(device) # Input: x_0 ... x_{T-1}
            targets = batch[:, 1:].to(device) # Target: x_1 ... x_T
            
            optimizer.zero_grad()
            
            # Forward pass: returns hidden states (B, T, Hidden)
            output = model(inputs)
            
            # We need a projection to vocab size for loss calculation
            # In a real rigorous implementation, this is often tied to embeddings, 
            # but here we'll assume the embedding layer transpose or a separate head is used.
            # For this homework, usually we add a final head. Let's add it dynamically or assume the model has it.
            # *Correction*: Transformer class returns 'output' (hidden states). 
            # We need logits. We can reuse the embedding matrix weight (tied embeddings) or a new linear layer.
            logits = torch.matmul(output, model.embedding.weight.t()) 

            # Flatten for CrossEntropy
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch} Step {i} Loss: {loss.item():.4f}")

        # Simple Dev Evaluation
        val_perp = evaluate_perplexity(model, dev_data, device)
        print(f"Epoch {epoch} Completed. Dev Perplexity: {val_perp:.4f}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def evaluate_perplexity(model, data, device='cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data:
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            
            output = model(inputs)
            logits = torch.matmul(output, model.embedding.weight.t())
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
            
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def generate(model, tokenizer, start_text, max_new_tokens=10, device='cpu'):
    model.eval()
    tokens = torch.tensor(tokenizer.encode(start_text)).unsqueeze(0).to(device) # (1, T)
    
    # TODO: Write a generation loop. This should generate up to
    # `max_new_tokens` tokens. We're doing greedy decoding, so for
    # every step, you should simply generate the most probable token
    # given the context. If the EOS token is generated,
    # break out of the loop early.
    # STUDENT START --------------------------------------
    for _ in range(max_new_tokens):
        with torch.no_grad():
            output = model(tokens) # (1, T, H)
            logits = torch.matmul(output, model.embedding.weight.t()) # (1, T, Vocab)
            
            # [cite_start]Greedy: take max of last token [cite: 95]
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0) # (1, 1)
            
            tokens = torch.cat((tokens, next_token), dim=1)
    # STUDENT END ------------------------------------------
    
    decoded = tokenizer.decode(tokens[0].tolist())
    print(f"Generated: {decoded}\n")
    return decoded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                        help='Options: TRANSFORMER, TRANSFORMER-MH, TRANSFORMER-MH, or path to model')
    parser.add_argument('--data', type=str, default="PILE", choices=["PILE", "WMT"])
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer() # Assumption from dataset.py
    vocab_size = tokenizer.vocab_size 
    
    # Model hyperparameters
    hidden_dim = 256
    context_len = 128 
    
    # Initialize Model based on flag
    if 'TRANSFORMER-MH' in args.model:
        model = MultiHeadTransformer(vocab_size, hidden_dim, context_len)
    elif 'TRANSFORMER' in args.model:
        model = Transformer(vocab_size, hidden_dim, context_len)
    else:
        model = MultiHeadTransformer(vocab_size, hidden_dim, context_len)
        model.load_state_dict(torch.load(args.model))

    # If WMT data is requested, reformat it as X =|= Y
    # We assume 'dataset.py' handles the internal formatting if we request 'WMT'
    train_data = load_data('train') if args.data == 'PILE' else get_translation_data('train')
    dev_data = load_data('dev') if args.data == 'PILE' else get_translation_data('dev')

    if args.evaluate_only:
        perp = evaluate_perplexity(model, dev_data, device)
        print(f"Perplexity: {perp}")
        
    elif 'generate' in args.model:
        prompts = ["Cats like to", "Yesterday, I went to the", "What is 5 + 5?"]
        for p in prompts:
            generate(model, tokenizer, p, device=device)
            
    else:
        train(model, train_data, dev_data, device=device, save_path=f"{args.model}_trained.pt")
        if args.data == 'WMT':
            pass
        else:
            print(f"Final Perplexity: {evaluate_perplexity(model, dev_data, device)}")

if __name__ == '__main__':
    main()