import numpy as np
import pandas as pd
import argparse
import pickle
import gc
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import sys
from transformers.modeling_outputs import BaseModelOutput

'''
decode the generated features to sequence using prot_t5_xl_uniref50 decoder
'''

def get_len(feature):
    pep_len = 0
    for i in range(len(feature)):
        if sum(abs(feature[i])) < 10:
            pep_len = i + 1
            break
    if pep_len == 0:
        pep_len = len(feature)
    return pep_len

def decode_seq(embed_feature, max_length, model, tokenizer):
    """decode embed_feature to sequence"""
    # set params
    encoder_last_hidden_state = torch.unsqueeze(
        torch.from_numpy(embed_feature), 0)
    encoder_output = BaseModelOutput(
        last_hidden_state=encoder_last_hidden_state,
        hidden_states=None,
        attentions=None,)
    model_kwargs = {"encoder_outputs": encoder_output}

    # generate seq from embeddings
    generated_tokens = model.generate(
        input_ids=None, bos_token_id=0,
        max_new_tokens=max_length, **model_kwargs)
    generated_seq = tokenizer.decode(
        generated_tokens[0], skip_special_tokens=True)
    
    generated_seq = generated_seq.replace(" ", "")   
    print(generated_seq)

    return generated_seq

def main():
    in_name = '/share/org/BGI/bgi_cshanyj/code/BiAMP/Text-to-AMP/generation/gen_emb_result/checkpoints_finetune.ema_0.9999_001000.pt.infill_control_no_generate_pep_emd.diff_step_200.num_samples_100.sd_2781084227.bsz_512.x_norm.z_norm.nocontrol.1.pickle'
    out_name = 'new_seq.txt'

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # input decoder model
    local_model = "../pretrained_models/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"
    tokenizer = T5Tokenizer.from_pretrained(local_model, do_lower_case=False)
    model = T5ForConditionalGeneration.from_pretrained(local_model)
    gc.collect()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()

    # input controllable generated samples (embeddings)
    input_pkl_dir = './generation/gen_emb_result'
    processed_pkl_dir = './improved_diffusion/gen_emb'
    feature_file = os.path.join(input_pkl_dir, in_name)
    feature_pkl = pd.read_pickle(feature_file)
    # data = torch.load(feature_file)
    # feature_pkl = data

    # get length of generated AMPs
    feature_len = list(map(get_len, list(feature_pkl['feature'])))
    feature_pkl["len"] = feature_len
    
    # decode feature to aa seq
    feature_seq = list(map(lambda x, y: decode_seq(x, y, model, tokenizer),
                           list(feature_pkl['feature']),
                           feature_pkl['len']))
    feature_pkl['sequence'] = feature_seq

    out_file = os.path.join(processed_pkl_dir, out_name)
    with open(out_file, "wb") as file:
        pickle.dump(feature_pkl, file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()












