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
        learn_sigma=False,                # 不学习方差 
        sigma_small=False,                
        class_cond=False,
        diffusion_steps=2000,             # 扩散时间步 
        timestep_respacing="",            # 跳过部分时间步的diffusion_steps的子集？
        noise_schedule="sqrt",            # 噪声权重 beta alpha 生成方式
        use_kl=False,                     # 损失函数是否使用 KL散度
        rescale_learned_sigmas=True,      # 损失函数是否使用 重新缩放的均方误差
        use_scale_shift_norm=True,       
        predict_xstart=True,              # predict_xstart为True, 模型预测的目标是X0
        rescale_timesteps=True,
        use_checkpoint=False,
        model_arch='skip-transformer',    # 根据 model_arch 选择不同的 Denoiser架构。
        in_channel=1024,
        out_channel=1024,
        training_mode='emb',              # 训练模式
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
    return model, diffusion

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
    # 获得噪声权重betas
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    # 确定损失函数类型 training_mode默认是emb
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
            loss_type = gd.LossType.RESCALED_MSE  # 损失函数是重新缩放的MSE
        else:
            loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        # model_mean_type决定模型预测的目标是噪声EPSILON还是初始数据START_X，默认是predict_xstart为True, 预测X0
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
