import os
import re
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import GPT2Config, GPT2Tokenizer, T5Tokenizer
from train import ProtT5_GPT2, initialize_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQ_LEN = 51
MAX_TEXT_LEN = 35
BATCH_SIZE = 32

CSV_PATH = "./prediction_result/generated.csv"
OUTPUT_PATH = "./prediction_result/predictions.csv"
SAVE_DIR = "./prot_gpt2_models"

local_prot_t5_path = "../pretrained_models/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"
local_gpt2_path = "../pretrained_models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

# Tokenizers
prot_tokenizer = T5Tokenizer.from_pretrained(local_prot_t5_path, do_lower_case=False)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(local_gpt2_path)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_tokenizer.bos_token = gpt2_tokenizer.eos_token

def rare__aa2X(seq):
    return re.sub(r"[UZOB]", "X", seq)

def add_blank(seq):
    return " ".join(list(seq))

def preprocess_sequence(seq):
    return add_blank(rare__aa2X(seq))

@torch.no_grad()
def generate_batch(model, sequences):
    processed = [preprocess_sequence(seq) for seq in sequences]

    encoded = prot_tokenizer(
        processed,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(device)

    encoder_outputs = model.encoder(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask
    )
    encoder_hidden_states = model.encoder_hidden_proj(encoder_outputs.last_hidden_state)

    decoder_input_ids = torch.full(
        (len(sequences), 1),
        gpt2_tokenizer.bos_token_id,
        device=device,
        dtype=torch.long
    )

    generated_ids = decoder_input_ids

    for _ in range(MAX_TEXT_LEN):
        outputs = model.decoder(
            input_ids=generated_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoded.attention_mask,
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)

        if (next_tokens == gpt2_tokenizer.eos_token_id).all():
            break

    texts = [
        gpt2_tokenizer.decode(ids, skip_special_tokens=True).strip()
        for ids in generated_ids
    ]
    return texts

if __name__ == "__main__":
    model = initialize_model()
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model_gpt2.pth", map_location=device))
    model.to(device)
    model.eval()

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise ValueError("The input CSV file is empty or the format is incorrect.")

    sequences = df.iloc[:, 0].tolist()
    results = []

    for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="Generating"):
        batch = sequences[i:i+BATCH_SIZE]
        try:
            batch_results = generate_batch(model, batch)
        except Exception as e:
            print(f"Batch creation could not be completed: {e}")
            batch_results = [""] * len(batch)
        results.extend(batch_results)

    # 保存
    df["Generated_Description"] = results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n Successfully generated {len(results)} sequence descriptions. Results saved to: {OUTPUT_PATH}")
