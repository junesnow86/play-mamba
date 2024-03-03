import math

import torch
from tqdm import tqdm

from data_utils import pad_sequences_3d

def train(model, tokenizer, data_loader, optimizer, criterion, device, 
          max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].clone()

        target = input_ids[:, 1:].to(device)
        input_ids = input_ids[:, :-1].to(device)

        input_ids = pad_sequences_3d(input_ids, pad_value=tokenizer.pad_token_id)
        target = pad_sequences_3d(target, max_len=input_ids.size(1), pad_value=tokenizer.pad_token_id)

        output = model(input_ids)
        loss = criterion(output, target)

        loss.backward()

        for name, param in model.named_parameters():
            if "out_proj.bias" not in name:
                # Clip gradients but not bias for out_proj
                torch.nn.utils.clip_grad_norm_(param, max_grad_norm)

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate(model, tokenizer, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].clone().detach().to(device)
        target = input_ids[:, 1:]
        input_ids = input_ids[:, :-1]

        input_ids = pad_sequences_3d(input_ids, pad_value=tokenizer.pad_token_id)
        target = pad_sequences_3d(target, max_len=input_ids.size(1), pad_value=tokenizer.pad_token_id)

        output = model(input_ids)
        loss = criterion(output, target)
        total_loss += loss.item()
    return total_loss / len(data_loader)

def calculate_perplexity(loss):
    return math.exp(loss)


if __name__ == "__main__":
    import os
    from transformers import AutoTokenizer
    from data_utils import load_enwiki8_dataset, encode_dataset, Enwiki8Dataset
    from model import Mamba

    batch_size = 256

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if os.path.exists("encoded_inputs_mamba.pt"):
        encoded_inputs = torch.load("encoded_inputs_mamba.pt")
    else:
        enwiki8_data = load_enwiki8_dataset()
        encoded_inputs = encode_dataset(tokenizer, enwiki8_data)
        torch.save(encoded_inputs, "encoded_inputs_mamba.pt")

    data = {"input_ids": encoded_inputs}

    # Split data
    total_size = len(data["input_ids"])
    train_size = int(0.8 * total_size)
    train_data = {key: val[:train_size] for key, val in data.items()}
    val_data = {key: val[train_size:] for key, val in data.items()}
    train_dataset = Enwiki8Dataset(train_data)
    val_dataset = Enwiki8Dataset(val_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mamba(seq_len=100, d_model=8, state_size=128).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

    for epoch in range(20):
        train_loss = train(model, tokenizer, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, tokenizer, val_loader, criterion, device)
        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Perplexity: {calculate_perplexity(val_loss):.4f}")
