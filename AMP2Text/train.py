import torch
import torch.nn as nn
import re
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5EncoderModel
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", buffering=1)  

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  

    def flush(self):
        self.terminal.flush()
        self.log.flush()
log = Logger("./output.txt")   

def rare__aa2X(sequence):
    return re.sub(r"[UZOB]", "X", sequence)
def add_blank(sequence):
    return " ".join(re.findall(".{1}", sequence))

MAX_SEQ_LEN = 51            
MAX_TEXT_LEN = 35
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 5
SAVE_DIR = "./AMP2Text/prot_gpt2_models"

local_prot_t5_path = "./pretrained_models/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"
local_gpt2_path = "./pretrained_models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

# Tokenizers
prot_tokenizer = T5Tokenizer.from_pretrained(local_prot_t5_path, do_lower_case=False)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(local_gpt2_path)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

config = GPT2Config.from_pretrained(local_gpt2_path)
config.add_cross_attention = True  

class AMP_Dataset(Dataset):
    def __init__(self, csv_path, prot_tokenizer, gpt2_tokenizer, max_length=MAX_SEQ_LEN):
        self.data = pd.read_csv(csv_path, encoding="utf-8")
        self.prot_tokenizer = prot_tokenizer
        self.gpt2_tokenizer = gpt2_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_seq = str(self.data.iloc[idx, 0])
        desc = self.data.iloc[idx, 1]
        processed_seq = add_blank(rare__aa2X(raw_seq))

        encoded_seq = self.prot_tokenizer(
            processed_seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

        input_ids = encoded_seq.input_ids.squeeze(0)
        attention_mask = encoded_seq.attention_mask.squeeze(0)

        encoded_desc = self.gpt2_tokenizer(
            desc,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LEN,
        )

        decoder_input_ids = encoded_desc.input_ids.squeeze(0)
        decoder_attention_mask = encoded_desc.attention_mask.squeeze(0)

        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask

class ProtT5_GPT2(nn.Module):
    def __init__(self, prot_encoder, gpt2_decoder):
        super().__init__()
        self.encoder = prot_encoder
        self.decoder = gpt2_decoder
        self.encoder_hidden_proj = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU()
        )

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_hidden_states = self.encoder_hidden_proj(encoder_hidden_states)

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            labels=labels,
        )

        return outputs.loss, outputs.logits

def initialize_model():
    prot_encoder = T5EncoderModel.from_pretrained(local_prot_t5_path)
    gpt2_decoder = GPT2LMHeadModel.from_pretrained(local_gpt2_path, config=config)

    gpt2_decoder.config.pad_token_id = gpt2_tokenizer.pad_token_id
    gpt2_decoder.resize_token_embeddings(len(gpt2_tokenizer))

    model = ProtT5_GPT2(prot_encoder, gpt2_decoder).to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False
    return model

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, decoder_input_ids, _ = [t.to(device) for t in batch]
            labels = decoder_input_ids.clone()
            loss, _ = model(input_ids, attention_mask, decoder_input_ids, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, decoder_input_ids, _ = [t.to(device) for t in batch]
            labels = decoder_input_ids.clone()

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, decoder_input_ids, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate_model(model, val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        log_msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
        log.write(log_msg)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_gpt2.pth")
            print("✅ Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹️ Early stopping triggered.")
                break

    torch.save(model.state_dict(), f"{SAVE_DIR}/final_model_gpt2.pth")
    print("📦 Final model saved.")

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/loss_curve_gpt2.png")
    plt.close()

# 主程序
if __name__ == "__main__":
    full_data = pd.read_csv("./data/description.csv", encoding="utf-8")  
    total_len = len(full_data)                      
    test_size = int(0.1 * total_len)                 
    val_size = int(0.1 * total_len)                  
    train_size = total_len - val_size - test_size   
    indices = torch.randperm(total_len).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    train_data = full_data.iloc[train_indices]
    val_data = full_data.iloc[val_indices]
    test_data = full_data.iloc[test_indices]
    test_data.to_csv("./data/test_set.csv", index=False)
    print("test_dataset has saved to ./data/test_set.csv")
    train_data.to_csv("./data/_train_tmp.csv", index=False)
    val_data.to_csv("./data/_val_tmp.csv", index=False)
    train_dataset = AMP_Dataset("./data/_train_tmp.csv", prot_tokenizer, gpt2_tokenizer)
    val_dataset = AMP_Dataset("./data/_val_tmp.csv", prot_tokenizer, gpt2_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    model = initialize_model()
    train_model(model, train_loader, val_loader)
    os.remove("./data/_train_tmp.csv")
    os.remove("./data/_val_tmp.csv")


