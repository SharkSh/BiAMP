import os
import sys
sys.path.append("./generation")
import torch, json
import torch.nn as nn
import argparse
from transformers import set_seed
from improved_diffusion.emd_datasets import get_train_test_loader
from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch.distributed as dist
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5EncoderModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'    

def main():
    args = create_argparser().parse_args()           
    set_seed(args.seed)                              
    dist_util.setup_dist()                           
    logger.configure(dir="./generation/train_log")        
    logger.log("creating model and diffusion and denoiser...")
    model, diffusion, prediction = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    logger.log('model_path: ', args.model_path)
    if args.model_path != None: 
        model.load_state_dict(torch.load(args.model_path))
    model.to(dist_util.dev())   
    prediction.to(dist_util.dev())
    pytorch_total_params = sum(p.numel() for p in model.parameters()) 
    logger.log(f'the parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    logger.log(f'saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    logger.log("creating data loader...")
    print('load data', '*'*50)
    train_dataloader, valid_dataloader = get_train_test_loader(args.train_ds_path, args.batch_size)
    logger.log("training...")
    TrainLoop(
        model=model,                              
        diffusion=diffusion,
        prediction=prediction,
        data=train_dataloader,                     
        class_weight=None,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,                               
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,            
        save_interval=args.save_interval,          
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,      
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=valid_dataloader,                
        eval_class_weight=None,
        eval_interval=args.eval_interval,          
    ).run_loop()  

    dist.destroy_process_group()

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",     
        lr=0.00005,                      
        weight_decay=0.0,
        lr_anneal_steps=1000,         
        batch_size=256,
        microbatch=-1,                 
        ema_rate="0.9999",              
        log_interval=100,               
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,                 
        fp16_scale_growth=1e-3,
        seed=101,
        gradient_clipping=-1.0,
        eval_interval=1000,
        checkpoint_path='./generation/checkpoints_finetune',               
        model_path='./generation/checkpoints_pretrain/ema_0.9999_100000.pt',                               
        train_ds_path='./generation/data/seq_text_len_gt.pt',   
        weighted=None,
    )
    text_defaults = dict(
        modality='pep_emd',
        model_arch='skip_transformer',
        emb_scale_factor=1.0, 
        noise_level=0.0,
        preprocessing_num_workers=1,
    )
    defaults.update(model_and_diffusion_defaults())  
    defaults.update(text_defaults)                    
    parser = argparse.ArgumentParser()               
    add_dict_to_argparser(parser, defaults)          
    return parser

if __name__ == "__main__":
    main()

