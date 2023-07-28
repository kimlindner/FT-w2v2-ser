#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --export=NONE

# set TF_ENABLE_ONEDNN_OPTS environment variable against floating-point round-off errors
export TF_ENABLE_ONEDNN_OPTS=0

# find the Python executable dynamically
PYTHON_EXEC=$(which python)

FOLDS=$1
NUM_EXPS=$2
EPOCHS=$3

srun $PYTHON_EXEC ../run_pretrain.py \
   --precision 16 \
   --datadir ../Dataset/emodb/wav \
   --labelpath ../Dataset/emodb/labels \
   --labeling_method hard \
   --saving_path pretrain/checkpoints/PTAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS} \
   --training_step 10000 \
   --save_top_k 1 \
   --wav2vecpath ../model/wav2vec_large.pt \

srun $PYTHON_EXEC ../cluster.py \
   --datadir ../Dataset/emodb/wav \
   --outputdir pretrain/checkpoints/PTAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}/clusters \
   --model_path pretrain/checkpoints/PTAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}/last.ckpt \
   --labelpath ../Dataset/emodb/labels \
   --model_type wav2vec \
   --sample_ratio 1.0 \
   --num_clusters "64,512,4096" \
   --wav2vecpath ../model/wav2vec_large.pt \

srun $PYTHON_EXEC ../run_second.py \
   --saving_path pretrain/checkpoints/PTAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}/second \
   --precision 16 \
   --datadir ../Dataset/emodb/wav \
   --labelpath pretrain/checkpoints/PTAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}/clusters/all-clus.json \
   --training_step 20000 \
   --warmup_step 100 \
   --save_top_k 1 \
   --lr 1e-4 \
   --batch_size 64 \
   --num_clusters "64,512,4096" \
   --use_bucket_sampler \
   --dynamic_batch \

srun $PYTHON_EXEC ../run_downstream_custom_multiple_fold.py \
   -- precision 16 \
   --num_exps $NUM_EXPS \
   --datadir ../Dataset/emodb/wav \
   --labeldir ../Dataset/emodb/labels \
   --pretrained_path pretrain/checkpoints/PTAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}/second \
   --outputfile downstream/checkpoints/PTAPT/folds${FOLDS}_numexps${NUM_EXPS}_epochs${EPOCHS}
