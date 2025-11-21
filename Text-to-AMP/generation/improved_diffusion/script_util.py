import torch
import torch.nn as nn
import os
import sys
import argparse
import inspect
from .transformer_model import MldDenoiser
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5EncoderModel
NUM_CLASSES = 1000

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=8,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,               
        sigma_small=False,                
        class_cond=False,
        diffusion_steps=2000,            
        timestep_respacing="",           
        noise_schedule="sqrt",            
        use_kl=False,                     
        rescale_learned_sigmas=True,      
        use_scale_shift_norm=True,       
        predict_xstart=True,            
        rescale_timesteps=True,
        use_checkpoint=False,
        model_arch='skip-transformer',    
        in_channel=1024,
        out_channel=1024,
        training_mode='emb',             
        vocab_size=66,
        config_name='bert-base-uncased',
        experiment_mode='lm',
        logits_mode=1,
    )

def create_model_and_diffusion(
    image_size,               
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,            
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    model_arch,
    in_channel,
    out_channel,
    training_mode,
    vocab_size,
    config_name,
    experiment_mode,
    logits_mode,
    **kwargs,
):
    model = create_model()
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,              
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        model_arch=model_arch,
        training_mode=training_mode,
    )
    prediction = initialize_model('./generation/checkpoints_pretrain/best_model.pth')
    return model, diffusion, prediction

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

def initialize_model(pretrained_prediction_path):
    local_prot_t5_path = "../pretrained_models/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"
    local_gpt2_path = "../pretrained_models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

    prot_tokenizer = T5Tokenizer.from_pretrained(local_prot_t5_path, do_lower_case=False)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(local_gpt2_path)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    config = GPT2Config.from_pretrained(local_gpt2_path)
    config.add_cross_attention = True  # cross-attention

    prot_encoder = T5EncoderModel.from_pretrained(local_prot_t5_path)
    gpt2_decoder = GPT2LMHeadModel.from_pretrained(local_gpt2_path, config=config)

    gpt2_decoder.config.pad_token_id = gpt2_tokenizer.pad_token_id
    gpt2_decoder.resize_token_embeddings(len(gpt2_tokenizer))

    model = ProtT5_GPT2(prot_encoder, gpt2_decoder)
    
    model.load_state_dict(torch.load(pretrained_prediction_path, map_location="cpu"))

    for param in model.encoder.parameters():
        param.requires_grad = False

    return model

def create_model(
    latent_dim=[1, 1024],
    num_layers=7,
    num_heads=4,
    dropout=0.1,
    condition="text",
    arch="trans_enc",
    text_encoded_dim=1024,
    nclasses=10,
    guidance_scale=7.5,
    guidance_uncondp=0.1,
):
    print("Creating model: skipTransformer")
    return MldDenoiser(
        ablation = {
            "SKIP_CONNECT": True,
            "PE_TYPE": "mld",
            "DIFF_PE_TYPE": "mld",
            "VAE_TYPE": "yes"
        },
        latent_dim=latent_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        condition=condition,
        arch=arch,
        text_encoded_dim=text_encoded_dim,
        nclasses=nclasses,
        guidance_scale=guidance_scale,
        guidance_uncondp=guidance_uncondp,
    )
    
def create_gaussian_diffusion(
    *,
    steps=1000,                         
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",            
    use_kl=False,
    predict_xstart=False,               
    rescale_timesteps=False,
    rescale_learned_sigmas=False,       
    timestep_respacing="",
    model_arch='conv-unet',
    training_mode='emb',
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if training_mode == 'e2e':
        if use_kl:
            loss_type = gd.LossType.E2E_KL
        else:
            loss_type = gd.LossType.E2E_MSE
    elif training_mode == 'e2e-simple':
        if use_kl:
            loss_type = gd.LossType.E2E_Simple_KL
        else:
            loss_type = gd.LossType.E2E_Simple_MSE
    else:
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL  
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE  
        else:
            loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        model_arch=model_arch,
        training_mode=training_mode,
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
