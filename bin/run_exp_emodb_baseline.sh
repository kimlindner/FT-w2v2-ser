#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --export=NONE

# set TF_ENABLE_ONEDNN_OPTS environment variable against floating-point round-off errors
export TF_ENABLE_ONEDNN_OPTS=0

# find the Python executable dynamically
PYTHON_EXEC=$(which python)

FOLDS=$1
NUM_EXPS=$2
EPOCHS=$3

srun $PYTHON_EXEC ../run_baseline_continueFT.py \
   --saving_path pretrain/checkpoints/TAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS} \
   --precision 16 \
   --datadir ../Dataset/emodb/wav \
   --labelpath ../Dataset/emodb/labels \
   --training_step 20000 \
   --warmup_step 100 \
   --save_top_k 1 \
   --lr 1e-4 \
   --batch_size 64\
   --use_bucket_sampler \

srun $PYTHON_EXEC ../run_downstream_custom_multiple_fold.py \
    --precision 16 \
    --max_epochs $EPOCHS \
    --num_exps $NUM_EXPS \
    --datadir  ../Dataset/emodb/wav \
    --labeldir ../Dataset/emodb/labels \
    --pretrained_path pretrain/checkpoints/TAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}/last.ckpt
    --saving_path downstream/checkpoints/TAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS} \
    --outputfile downstream/checkpoints/TAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}/metrics_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}.txt
