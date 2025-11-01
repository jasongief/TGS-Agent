#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=1
MASTER_PORT=6666
RANK=0

llama2_ckpt_path="./pretrained_weights/Llama-2-7b-chat-hf"

# Training Arguments
LOCAL_BATCH_SIZE=8 # no use
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05


# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=ref_avs_real
RUN_NAME=ref_avs_exp_ori_correct
OUTP_DIR=results
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'

pretrained_ckpt_base_dir=./results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05

python scripts/finetune/inference_hyper_lora.py \
    --llm_name llama \
    --model_name_or_path $llama2_ckpt_path \
    --freeze_backbone True \
    --lora_enable True \
    --use_hyper_lora False \
    --use_process True \
    --bits 32 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --ckpt_dir $pretrained_ckpt_base_dir/checkpoint-551 \
    --avqa_task False \
    --ave_task False \
    --avvp_task False \
    --arig_task False \
    --avcap_task False \
    --ms3_task False \
    --s4_task False \
    --avss_task False \
    --ref_avs_task True \
    --avs_ckpt_dir $pretrained_ckpt_base_dir \
    --test_name test_u \
    --device cuda:3 \
    --multi_frames False \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path "./pretrained_weights/clip-vit-large-patch14" \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path "./pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"  \
    --audio_query_token_nums 32 \
    --seg_branch False \
    --prompt_embed_dim 256 \
    --mask_decoder_transformer_depth 2 \
    --low_res_mask_size 112 \
    --image_scale_nums 2 \
    --token_nums_per_scale 3 \
    --avs_query_num 300 \
    --num_classes 1 \
    --query_generator_num_layers 2 \
    --output_dir 'test'

