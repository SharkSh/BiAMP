import gc
import os
import re
import torch
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def rare__aa2X(sequence):
    return re.sub(r"[UZOB]", "X", sequence)
def add_blank(sequence):
    aa_list = re.findall(".{1}", sequence)
    new_aa = " ".join(aa_list)
    return new_aa

def ProtT5_embed(seq, tokenizer, model):
    ids = tokenizer.batch_encode_plus(
        seq, 
        add_special_tokens=True, 
        padding="max_length",
        max_length=51,    
        truncation=True,
        return_tensors="pt"
    )
    input_ids = ids['input_ids'].to(device)
    attention_mask = ids['attention_mask'].to(device)
    print("ProtT5_token_ids")
    print(input_ids)
    with torch.no_grad():
        embedding = model(input_ids, attention_mask=attention_mask).last_hidden_state
    print("ProtT5 embedding shape:", embedding.shape)  # [batch_size, 51, 1024]
    print("ProtT5 attention_mask shape:", attention_mask.shape)  # [batch_size, 51]
    return embedding, attention_mask

def T5_embed(text, tokenizer, model, max_len=32):
    if isinstance(text, pd.Series):
        text = text.tolist()
    text = [str(t) for t in text] 
    token_lengths = [len(tokenizer.encode(t, add_special_tokens=True)) for t in text]
    truncated_count = sum([l > max_len for l in token_lengths])
    if truncated_count > 0:
        print(f"⚠️ Warning: {truncated_count}/{len(text)} samples were truncated to max_length={max_len}.")
    else:
        print("✅ No truncation occurred in this batch.")
    ids = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = ids['input_ids'].to(device)
    attention_mask = ids['attention_mask'].to(device)
    print("T5_token_ids")
    print(input_ids)
    with torch.no_grad():
        embedding = model(input_ids, attention_mask=attention_mask).last_hidden_state
    print("T5 embedding shape:", embedding.shape)  # [batch_size, seq_len, hidden_dim]
    return embedding

def gpt2_get_token(desc, gpt2_tokenizer, max_gt_len = 30):
    encoded_desc = gpt2_tokenizer(
        desc,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_gt_len,
    )
    ids = encoded_desc.input_ids.squeeze(0)
    return ids

def main():
    batch_size = 1024

    ProtT5_local_model = "../pretrained_models/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"
    ProtT5_tokenizer = T5Tokenizer.from_pretrained(ProtT5_local_model, do_lower_case=False, local_files_only=True)
    ProtT5_encoder = T5EncoderModel.from_pretrained(ProtT5_local_model, local_files_only=True).to(device)

    T5_local_model = "../pretrained_models/models--t5-Large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a"
    T5_tokenizer = T5Tokenizer.from_pretrained(T5_local_model, local_files_only=True)
    T5_encoder = T5EncoderModel.from_pretrained(T5_local_model, local_files_only=True).to(device)

    gpt2_local_model = "../pretrained_models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_local_model)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    df = pd.read_csv('./data/seq_text_len_gt.csv')

    df['processed_sequence'] = df['seq'].apply(rare__aa2X).apply(add_blank)
    len_sample = len(df['seq'])

    all_seq_embs = []
    all_text_embs = []
    all_len = []
    all_attention_mask = []
    all_ids = []

    for i in range(0, len_sample, batch_size):
        batch_seq = df['processed_sequence'][i: i + batch_size]
        batch_text = df['text'][i: i + batch_size]
        batch_len = df['length'][i: i + batch_size]
        batch_gt = df['gt'][i: i + batch_size].tolist()

        batch_seq_emb, batch_attention_mask = ProtT5_embed(batch_seq, ProtT5_tokenizer, ProtT5_encoder)
        batch_text_emb = T5_embed(batch_text, T5_tokenizer, T5_encoder, max_len = 32)
        batch_length = T5_embed(batch_len, T5_tokenizer, T5_encoder, max_len = 2)[:, 0, :].unsqueeze(dim = 1)
        batch_ids = gpt2_get_token(batch_gt, gpt2_tokenizer)

        all_seq_embs.append(batch_seq_emb.cpu())
        all_text_embs.append(batch_text_emb.cpu())
        all_len.append(batch_length.cpu())
        all_attention_mask.append(batch_attention_mask.cpu())
        all_ids.append(batch_ids.cpu())
        print(f"Batch {i//batch_size} processed.")

        del batch_seq_emb, batch_attention_mask, batch_text_emb, batch_length, batch_ids
        torch.cuda.empty_cache()
        gc.collect()

    final_seq_emb = torch.cat(all_seq_embs, dim=0)
    final_text_emb = torch.cat(all_text_embs, dim=0)
    final_len = torch.cat(all_len, dim=0)
    final_mask = torch.cat(all_attention_mask, dim=0)
    final_ids = torch.cat(all_ids, dim = 0)

    print(f"Shape of final_seq_emb: {final_seq_emb.shape}")
    print(f"Shape of final_text_emb: {final_text_emb.shape}")
    print(f"Shape of final_len: {final_len.shape}")
    print(f"Shape of final_mask: {final_mask.shape}")
    print(f"Shape of final_ids: {final_ids.shape}")

    torch.save(
        {"seq_emb": final_seq_emb,          # [batch, 51, 1024]
        "mask": final_mask,                 # [batch, 51]
        "text_emb": final_text_emb,         # [batch, 32, 1024]
        "length": final_len,                # [batch, 1024]          
        "ids": final_ids},                  # [batch, 30]    
        os.path.join('./data/', "seq_text_len_gt.pt")
    )
    print("All batches processed and final embeddings saved.")

if __name__ == "__main__":
    main()
