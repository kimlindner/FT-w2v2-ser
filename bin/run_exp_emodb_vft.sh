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

srun $PYTHON_EXEC ../run_downstream_custom_multiple_fold.py \
    --precision 16 \
    --max_epochs $EPOCHS \
    --num_exps $NUM_EXPS \
    --datadir  ../Dataset/emodb/wav \
    --labeldir ../Dataset/emodb/labels \
    --saving_path downstream/checkpoints/VFT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS} \
    --outputfile downstream/checkpoints/VFT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}/metrics_folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}.txt
