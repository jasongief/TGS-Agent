import os,sys
sys.path.append(os.getcwd())
from os.path import join,exists
import pathlib
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')
from ipdb import set_trace

from torch.utils.data import DataLoader
import transformers
from torch.cuda.amp import autocast

from configs.unified_config import ModelArguments,DataArguments,TrainingArguments,InferenceArguments

from dataset.unified_dataset import get_dataset_collator
from utils.util import set_seed,find_all_linear_names,prepare_sample,write2json,load_ckpt
from utils.avss_utils import (
    mask_iou,compute_miou_from_jsonl,calc_color_miou_fscore,
    save_color_mask,save_gt_mask,Eval_Fmeasure,
    metric_s_for_null
)
from utils.deepspeed_utils import *

local_rank = None




def inference_cot_ref_avs(dataloader,ckpt_dir,model,tokenizer,test_name='test_s'):
    save_dir = join(ckpt_dir,f'inference_cot_ref_avs_{test_name}_bs')
    os.makedirs(save_dir,exist_ok=True)
    pbar = tqdm(total=len(dataloader),desc=f'inference cot ref-avs {test_name}')
    fp = join(save_dir,f'inference_results.jsonl')
    for step,sample in enumerate(dataloader):
        batch_metadata = sample.pop('batch_metadata')
        bs = len(batch_metadata)
        sample = prepare_sample(data=sample)
        sample.update(
            {
                'use_cache':True,
                'max_new_tokens':256,
            }
        )
        with torch.no_grad():
             with autocast(dtype=torch.bfloat16):
                output = model.generate(
                    **sample,
                    temperature=0.3,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                output = tokenizer.batch_decode(output,skip_special_tokens=False)

        for i in range(bs):
            uid = batch_metadata[i]['uid']
            ref = batch_metadata[i]['ref']
            predict = output[i]
            metadata = {
                'uid':uid,
                'ref':ref,
                'predict':predict,
            }
            write2json(fp=fp,dict_data=metadata)

        pbar.update(1)
    pbar.close()



def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, InferenceArguments))
    model_args, data_args, training_args, infer_args = parser.parse_args_into_dataclasses()

    if model_args.llm_name == 'llama':
        d_model = 4096
    elif model_args.llm_name == 'qwen':
        d_model = 3584

    local_rank = training_args.local_rank
    compute_dtype = torch.float32
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16
    
    pretrain_model_name_or_path = model_args.model_name_or_path
    if model_args.llm_name == 'llama':
        from models.unified_llama import UnifiedForCausalLM
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config=config,
            torch_dtype=compute_dtype
        )
    elif model_args.llm_name == 'qwen':
        from models.unified_qwen import UnifiedForCausalLM
        from transformers import Qwen2Config
        config = Qwen2Config.from_pretrained(pretrain_model_name_or_path, local_files_only=True)
        config._attn_implementation = attn_implementation
        model = UnifiedForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config = config,
            torch_dtype = compute_dtype
        )

    model.config.use_cache = True 

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        if training_args.use_hyper_lora:
            from peft_hyper import LoraConfig,get_peft_model
            lora_trainable="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
            target_modules = lora_trainable.split(',')
            lora_rank = 8
            lora_alpha = 16
            lora_dropout = 0.05
            lora_nums = 3
            modules_to_save = None
            peft_config = LoraConfig(
                task_type = "CAUSAL_LM",
                target_modules = target_modules,
                inference_mode = False,
                r = lora_rank, 
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                lora_nums = lora_nums,
                # modules_to_save=modules_to_save
            )
            model = get_peft_model(model, peft_config)
        else:
            from peft import LoraConfig, get_peft_model
            lora_trainable = "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj"
            target_modules = lora_trainable.split(',')

            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)


    if model_args.llm_name == 'qwen':
        from transformers import Qwen2Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
    
    elif model_args.llm_name == 'llama':
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            pretrain_model_name_or_path,
            padding_side="left",
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ori_tokenizer_vocab_nums = len(tokenizer)
    model.get_model().pad_token_id = tokenizer.pad_token_id
    model.get_model().init_multimodal_modules(visual_branch=training_args.visual_branch,
                                              audio_branch=training_args.audio_branch,
                                              segment_branch=training_args.seg_branch,
                                              d_model=d_model,vit_ckpt_path=model_args.vit_ckpt_path,
                                              select_layer_list=model_args.select_layer_list,
                                              select_feature=model_args.select_feature,
                                              image_size=model_args.image_size,
                                              patch_size=model_args.patch_size,
                                              visual_query_token_nums=model_args.visual_query_token_nums,
                                              audio_query_token_nums=model_args.audio_query_token_nums,
                                              BEATs_ckpt_path=model_args.BEATs_ckpt_path,
                                              prompt_embed_dim=model_args.prompt_embed_dim,
                                              mask_decoder_transformer_depth=model_args.mask_decoder_transformer_depth,
                                              low_res_mask_size=model_args.low_res_mask_size,
                                              avs_query_num=model_args.avs_query_num,
                                              num_classes=model_args.num_classes,
                                              query_generator_num_layers=model_args.query_generator_num_layers,
                                              dice_loss_weight=training_args.dice_loss_weight,
                                              bce_loss_weight=training_args.bce_loss_weight,
                                              use_vqgan=False)

    model.initialize_MM_tokenizer(tokenizer, use_vqgan=False)
    MM_tokenizer_vocab_nums = len(tokenizer)
    print('ori_tokenizer_vocab_nums: ',ori_tokenizer_vocab_nums, ' MM_tokenizer_vocab_nums: ',MM_tokenizer_vocab_nums)


    infer_avs = False
    ckpt_dir = infer_args.ckpt_dir
    avs_ckpt_dir = infer_args.avs_ckpt_dir
    if not infer_avs:
        ckpt_path = join(ckpt_dir,'finetune_weights.bin')
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt,strict=False)
        print(f'load ckpt from {ckpt_path} finished...')

        nolora_ckpt_path = join(avs_ckpt_dir,'non_lora_trainables.bin')
        nolora_ckpt = torch.load(nolora_ckpt_path,map_location='cpu')
        model.load_state_dict(nolora_ckpt,strict=False)
        print(f'load ckpt from {nolora_ckpt_path} finished...')
    else:
        ## hyper lora ckpt
        ckpt_path = join(ckpt_dir,'finetune_weights.bin')
        ckpt = torch.load(ckpt_path,map_location='cpu')
        model.load_state_dict(ckpt,strict=False)
        print(f'load hyper_lora weights from {ckpt_path} finished...')
        ## seg module ckpt
        ckpt_path = join(avs_ckpt_dir,'finetune_weights.bin')
        ckpt = torch.load(ckpt_path,map_location='cpu')
        model.load_state_dict(ckpt,strict=False)
        print(f'load seg_module ckpt from {ckpt_path} finished...')

    device = infer_args.device
    torch.cuda.set_device(device)
    model.to(device)
    model.eval()
    if training_args.bf16:
        model.to(torch.bfloat16)
    
    image_processor = model.get_model().visual_encoder.image_processor if training_args.visual_branch else None
    dataset, collator = get_dataset_collator(data_args=data_args, tokenizer=tokenizer, 
                                             image_processor=image_processor,mode='test',
                                             test_name=infer_args.test_name)
    
    batch_size = 1 if infer_avs else 8 # fixed 8 in testing
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False,collate_fn=collator,drop_last=False)
    
    if data_args.ref_avs_task:
        test_name = infer_args.test_name
        inference_cot_ref_avs(dataloader=dataloader,ckpt_dir=ckpt_dir,model=model,tokenizer=tokenizer,test_name=test_name)
    print('inference finished...')

    
if __name__ == "__main__":
    train()

