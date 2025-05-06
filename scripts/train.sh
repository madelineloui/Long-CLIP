#!/bin/bash
#SBATCH --gres=gpu:volta:1

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate ovdsat

EXP_NAME="auto"
EPOCHS=25
BATCH_SIZE=2
LR=1e-6
WD=1e-2
LOG_SCALE=4.6052
WARMUP=200

python train/train.py \
  --lr $LR \
  --weight_decay $WD \
  --log_scale $LOG_SCALE \
  --exp_name $EXP_NAME \
  --warmup_length $WARMUP \
  --batch-size $BATCH_SIZE \
  --epochs $EPOCHS \
  --download-root ""