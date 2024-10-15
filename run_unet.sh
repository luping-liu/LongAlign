#!/bin/bash

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="xxx"

pkill wandb

echo $1 $2
if [ $1 = 'align' ]; then
  export MY_ENVIRON_SAMPLE='unipc'
  export MY_ENVIRON_DROP='0.1'
  accelerate launch --config_file ./acc_config.yaml \
  --num_processes 6 unet_align_ct5.py \
  --pretrained_decoder="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --mixed_precision="bf16" \
  --token_length 240 \
  --resolution 512 \
  --batch_size 16 \
  --accumulation_steps 2 \
  --learning_rate 3e-5 \
  --loss_type "l2" \
  --max_train_epochs 2 \
  --validation_steps 150 \
  --save_steps 1250 \
  --output_dir "results/sd15-align-$2" \
  --ckpt_dir "" \
  --resume
#  --debug

elif [ $1 = 'lcm' ]; then
  export MY_ENVIRON_SAMPLE='lcm'
  export MY_ENVIRON_DROP='0.0'
  accelerate launch --config_file ./acc_config.yaml \
  --num_processes 6 unet_lcm.py \
  --pretrained_decoder="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --mixed_precision="bf16" \
  --token_length 240 \
  --resolution 512 \
  --batch_size 16 \
  --accumulation_steps 2 \
  --learning_rate 3e-5 \
  --loss_type "huber" \
  --max_train_epochs 2 \
  --validation_steps 150 \
  --save_steps 1250 \
  --output_dir "results/sd15-lcm-$2" \
  --ckpt_dir "results/sd15-align-ct5f/" \
  --resume
#  --debug

elif [ $1 = 'reward' ]; then
  export MY_ENVIRON_SAMPLE='unipc'
  export MY_ENVIRON_DROP='0.0'
  export REWEIGHT=$3
  accelerate launch --config_file ./acc_config.yaml \
  --num_processes 6 unet_reward.py \
  --pretrained_decoder="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --mixed_precision="bf16" \
  --token_length 240 \
  --resolution 512 \
  --batch_size 8 \
  --accumulation_steps 1 \
  --learning_rate 3e-5 \
  --loss_type "l2" \
  --max_train_epochs 1 \
  --validation_steps 150 \
  --save_steps 1250 \
  --output_dir "results/sd15-reward-$2" \
  --ckpt_dir "results/sd15-align-ct5f/" \
  --resume
#  --debug

elif [ $1 = 'reward-lcm' ]; then
  export MY_ENVIRON_SAMPLE='unipc'
  export MY_ENVIRON_DROP='0.0'
  export REWEIGHT=$3
  accelerate launch --config_file ./acc_config.yaml \
  --num_processes 6 unet_reward_lcm.py \
  --pretrained_decoder="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --mixed_precision="bf16" \
  --token_length 240 \
  --resolution 512 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --learning_rate 3e-5 \
  --loss_type "l2" \
  --max_train_epochs 1 \
  --validation_steps 150 \
  --save_steps 1250 \
  --output_dir "results/sd15-rewardlcm-$2" \
  --ckpt_dir "results/sd15-align-ct5f/, results/sd15-lcm-ct5f/" \
  --resume
#  --debug

fi
