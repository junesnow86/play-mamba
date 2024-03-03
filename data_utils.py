import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Enwiki8Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.data.items()}

def pad_sequences_3d(sequences, max_len=None, pad_value=0):
    batch_size, seq_len, feature_size = sequences.shape
    if max_len is None:
        max_len = seq_len + 1
    padded_sequences = torch.full((batch_size, max_len, feature_size), fill_value=pad_value,
                                  dtype=sequences.dtype, device=sequences.device)
    padded_sequences[:, :seq_len, :] = sequences
    return padded_sequences

def load_enwiki8_dataset():
    print("Downloading enwiki8 dataset...")
    url = "http://mattmahoney.net/dc/enwik8.zip"
    import urllib
    urllib.request.urlretrieve(url, "enwik8.zip")
    from zipfile import ZipFile
    with ZipFile("enwik8.zip", "r") as zip_ref:
        data = zip_ref.read("enwik8").decode("utf-8")
    print("Done.")
    return data

def encode_dataset(tokenizer, text_data, max_length=100, d_model=8):
    def batch_encode(tokenizer, text_data, batch_size=256):
        batched_input_ids = []
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i + batch_size]
            inputs = tokenizer(batch,
                               add_special_tokens=True,
                               truncation=True,
                               padding="max_length",
                               max_length=max_length,
                               return_tensors="pt")
            batched_input_ids.append(inputs["input_ids"])
        return torch.cat(batched_input_ids)

    # Assuming enwiki8_data is a list of sentences
    input_ids = batch_encode(tokenizer, text_data)

    vocab_size = len(tokenizer.vocab)

    embedding_layer = nn.Embedding(vocab_size, d_model)

    def batch_embedding_call(input_ids, embedding_layer, batch_size=256):
        num_batches = input_ids.size(0) // batch_size
        output_embeddings = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            input_ids_batch = input_ids[start_idx:end_idx]
            with torch.no_grad():
                batch_embeddings = embedding_layer(input_ids_batch)
            output_embeddings.append(batch_embeddings)
        return torch.cat(output_embeddings, dim=0)

    encoded_inputs = batch_embedding_call(input_ids, embedding_layer, batch_size=1).float()
    return encoded_inputs
