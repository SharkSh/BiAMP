"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import time
import argparse
import os, json, sys
sys.path.append("./generation")
import stanza
import spacy_stanza
import numpy as np
import pickle
import torch as th
from transformers import set_seed
import torch.distributed as dist
from transformers import T5Tokenizer, T5EncoderModel
from improved_diffusion.test_util import get_weights, denoised_fn_round
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from infill_util import langevin_fn3, langevin_fn3_mod, langevin_fn3_nocontrol, get_score, langevin_fn3_compose, langevin_fn1, langevin_fn4, langevin_fn_tree, langevin_fn_length

def T5_embed(text, tokenizer, model, max_len):
    ids = tokenizer(
            text, 
            add_special_tokens=True, 
            padding="max_length", 
            max_length=max_len,
            truncation=True, 
            return_tensors="pt"
            )
    input_ids = ids['input_ids']
    attention_mask = ids['attention_mask']
    with th.no_grad():
        embedding = model(input_ids, attention_mask=attention_mask).last_hidden_state
    return embedding  # [B, 32, 1024]

def main():

    args = create_argparser().parse_args()
    sd_tmp = args.seed
    set_seed(args.seed)
    model_path_tmp = args.model_path
    print('seed: ', args.seed)
    print('m (img): ', args.m)
    print('n (noise): ', args.n)

    config_path = './generation/checkpoints_finetune/training_args.json'
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)
    # 
    args.noise_level = 0.0
    args.sigma_small = True
    args.seed = sd_tmp
    args.model_path = model_path_tmp
    print('model path: ', args.model_path)
    args.batch_size = 2 * args.batch_size

    if args.eval_task_.startswith('control_'):
        args.diffusion_steps = 200  # 500
        args.eta = 1.0

    print('batch size: ', args.batch_size)
    print('eta: ', args.eta)
    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')
    logger.log("creating model and diffusion...")
    model, diffusion, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()

    if args.modality in ['pep_emd']:
        todo_pad_token = -1   
        pad_token = 0         
        if args.eval_task_.startswith('control'):
 
            right_pad = th.empty(args.pep_len).fill_(pad_token).long()   
            encoded_partial_seq = [th.cat([right_pad], dim=0)]            
            encoded_partial_seq[0][0] = 0
            partial_seq = []
            
            if args.eval_task_ == 'control_no':  
                control_label_lst = [1]
                control_constraints = []
                for label_class in control_label_lst:
                    label_ids = th.tensor(label_class).unsqueeze(0)
                    debug_lst = []
                    langevin_fn_selected = partial(langevin_fn3_nocontrol, debug_lst, label_ids.expand(args.batch_size, -1), 0.1)
                    control_constraints.append((langevin_fn_selected, label_class))

                partial_seq = control_constraints
                # print('control_constraints', control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-'*20)

    logger.log("sampling...")
    sample_dict = {}
    if True:
        print('encoded_partial_seq: ', encoded_partial_seq)    
        print('partial_seq: ', partial_seq)          
        for (encoded_seq_ori, control_helper) in zip(encoded_partial_seq, partial_seq):
            all_images = []   
            all_init = []     
            all_progress = [] 
            all_labels = []   
            print(args.num_samples, encoded_seq_ori.shape, 'encoded_seq_ori.shape')
            print('encoded_seq_ori: ', encoded_seq_ori)
            epoch=0
            while len(all_images) * args.batch_size < args.num_samples: 
                model_kwargs = {}              
                encoded_seq = encoded_seq_ori.clone()
                encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size,-1)   
                partial_mask_temp = (encoded_seq == todo_pad_token).view(args.batch_size, -1)
                encoded_seq.masked_fill_(encoded_seq == todo_pad_token, 3)          
                encoded_seq_hidden = encoded_seq.cuda()

                seqlen = encoded_seq.size(1)
                partial_mask = partial_mask_temp.unsqueeze(-1).expand(-1, -1, args.in_channel)
                sample_shape = (args.batch_size, seqlen, args.in_channel, )

                if args.eval_task_.startswith('control'):
                    langevin_fn_selected, label_class_attributes = control_helper
                    print('label_class_attributes: ', label_class_attributes)
                    print('-*'*200, label_class_attributes, '-*'*200)
                    if args.use_ddim:       
                        loop_func_ = diffusion.ddim_sample_loop_progressive
                    else:
                        loop_func_ = diffusion.p_sample_loop_progressive

                    sequence = "Develop a peptide sequence."
                    length = "28"

                    T5_path = "../pretrained_models/models--t5-Large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a"
                    T5_tokenizer = T5Tokenizer.from_pretrained(T5_path, local_files_only=True)
                    T5_encoder = T5EncoderModel.from_pretrained(T5_path, local_files_only=True)
                    text_emb = T5_embed(sequence, T5_tokenizer, T5_encoder, 32).to(encoded_seq_hidden.device)
                    untext_emb = th.zeros_like(text_emb).to(encoded_seq_hidden.device)
                    text_emb = text_emb.expand(int(args.batch_size/2), -1, -1)     
                    untext_emb = untext_emb.expand(int(args.batch_size/2), -1, -1) 
                    text_all = th.cat([untext_emb, text_emb], dim=0)  # (batch_size, 32, 1024)

                    len_emb = T5_embed(length, T5_tokenizer, T5_encoder, 2)[:, 0, :].unsqueeze(dim = 1).to(encoded_seq_hidden.device)
                    length = len_emb.expand(int(args.batch_size/2), -1, -1) 
                    uncond_length = th.zeros_like(length)
                    length_all = th.cat([uncond_length, length], dim=0)  # shape: (batch_size, 1, 1024)
                    model_kwargs = {"text_emb": text_all, "length": length_all}

                    for sample in loop_func_(          # diffusion.ddim_sample_loop_progressive
                            model,
                            sample_shape,
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,       # 传参
                            device=encoded_seq_hidden.device,
                            langevin_fn=None,
                            eta=args.eta,
                            n=args.n,
                            m=args.m
                    ):
                        final = {}
                        final["sample"] = sample["sample"]
                        final["init"] = sample["init"].cuda()
                        final["progress"] = sample["progress"].cuda()
                        print("final['progress'].shape: ", final['progress'].shape)

                sample = final
                epoch+=1
                print('epoch: ', epoch)
    
                scale = 0.9     
                batch_size_2 = sample["sample"].shape[0] // 2
                noise_pred_uncond = sample["sample"][:batch_size_2]
                noise_pred_cond = sample["sample"][batch_size_2:]
                sample["sample"] = scale * noise_pred_cond + (1 - scale) * noise_pred_uncond   
                sample["sample"] = sample["sample"].contiguous() 
                gathered_samples = [th.zeros_like(sample["sample"]) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample["sample"])  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

                gathered_init = [th.zeros_like(sample["init"]) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_init, sample["init"])  # gather not supported with NCCL
                all_init.extend([sample.cpu().numpy() for sample in gathered_init])
                print('all_init: ', all_init)

                gathered_progress = [th.zeros_like(sample["progress"]) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_progress, sample["progress"])  # gather not supported with NCCL
                all_progress.extend([sample.cpu().numpy() for sample in gathered_progress])
                print('all_progress: ', all_progress)

                if args.class_cond:
                    gathered_labels = [
                        th.zeros_like(classes) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_labels, classes)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                logger.log(f"created {len(all_images) * args.batch_size} samples")
    
            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]
            arr_init = np.concatenate(all_init, axis=0)
            arr_init = arr_init[: args.num_samples]
            arr_progress = np.concatenate(all_progress, axis=1)
            arr_progress = arr_progress[:, : args.num_samples, :, :]

            print('arr.shape: ', arr.shape)
            print('arr_init.shape: ', arr_init.shape)
            print('arr_progress.shape: ', arr_progress.shape)
            if args.verbose == 'pipe':
                sample_dict[str(label_class_attributes)] = {}
                sample_dict[str(label_class_attributes)]["feature"] = arr
                sample_dict[str(label_class_attributes)]["init"] = arr_init
                sample_dict[str(label_class_attributes)]["progress"] = arr_progress
                # print(f'writing to sample_dict, for class {" ".join(label_class_attributes)}')
                print(f'writing to sample_dict, for class {str(label_class_attributes)}')

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'

    dist.barrier()
    logger.log("sampling complete")

    if args.verbose == 'pipe':
        print(f'sampled for {len(sample_dict)} control tasks')
        for label_class in sample_dict:
            print('seed: ', args.seed)
            if args.m == 0:
                x_distribution = 'norm'
            else:
                x_distribution = 'uni' + str(args.m)
            if args.n == 0:
                z_distribution = 'norm'
            else:
                z_distribution = 'uni' + str(args.n)
                
            print('x_distribution: ', x_distribution)
            print('z_distribution: ', z_distribution)
            
            out_path_pipe = os.path.join(
                args.out_dir,
                f"{model_base_name}.infill_{args.eval_task_}_{args.notes}.diff_step_{args.diffusion_steps}.num_samples_{args.num_samples}.sd_{args.seed}.bsz_{args.batch_size}.x_{x_distribution}.z_{z_distribution}.nocontrol.{label_class}.pickle")
        
            with open(out_path_pipe, 'wb') as fout:
                pickle.dump(sample_dict[label_class], fout)
                fout.close()
            
        out_path2 = out_path_pipe
        
    args.out_path2 = out_path2
    return args


def create_argparser():
    defaults = dict(
        data_dir="", 
        clip_denoised=False, 
        use_ddim=True, 
        eta=1.0, 
        num_samples=100, 
        batch_size=100, 
        model_path="./generation/checkpoints_finetune/ema_0.9999_001000.pt",   
        out_dir="./generation/gen_emb_result", 
        emb_scale_factor=1.0, 
        split='train', 
        debug_path='', 
        eval_task_='control_no',
        partial_seq="", 
        partial_seq_file="", 
        verbose='pipe', 
        tgt_len=15, 
        t_merge=100, 
        interp_coef=0.5, 
        notes='generate_pep_emd',
        start_idx=0, 
        end_idx=0, 
        pep_len=51, 
        # seed=102, 
        seed = int(time.time() * 1000) % (2**32),
        m=0.0, 
        n=0.0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    args = main()
    

