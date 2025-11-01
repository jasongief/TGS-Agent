#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=4
MASTER_PORT=6666
RANK=0

llama2_ckpt_path="./pretrained_weights/Llama-2-7b-chat-hf"

# Training Arguments
NUM_TRAIN_EPOCHS=6
LEARNING_RATE=1e-4
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

export RUN_NAME="epochs${NUM_TRAIN_EPOCHS}_lr${LEARNING_RATE}_bs${LOCAL_BATCH_SIZE}_gradacc${GRADIENT_ACCUMULATION_STEPS}_lora_r${LORA_R}alpha${LORA_ALPHA}dropout${LORA_DROPOUT}"
OUTP_DIR=results_real_tmp_for_reproduce

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY=""
export WANDB_PROJECT=results_real_tmp_for_reproduce
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'


torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/finetune/finetune_hyperlora.py \
    --deepspeed deepspeed/stage2-offload.json \
    --llm_name llama \
    --model_name_or_path $llama2_ckpt_path \
    --exp_desc "exp" \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --pretrain_ckpt_dir "./pretrained_weights/" \
    --avqa_task False \
    --ave_task False \
    --avvp_task False \
    --arig_task False \
    --avcap_task False \
    --ms3_task False \
    --s4_task False \
    --avss_task False \
    --ref_avs_task True \
    --save_modules vl_projector,al_projector,lora \
    --multi_frames False \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path "./pretrained_weights/clip-vit-large-patch14" \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path "./pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
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
    --ce_loss_weight 1.0 \
    --dice_loss_weight 0.5 \
    --bce_loss_weight 1.0 \
    --output_dir $OUTP_DIR/$WANDB_PROJECT/$RUN_NAME \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ddp_find_unused_parameters True \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps -1 \
    --save_total_limit 10 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --half_precision_backend "auto" \
    --dataloader_num_workers 4 \
    --report_to all \

