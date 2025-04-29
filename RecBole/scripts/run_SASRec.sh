#!/bin/bash

#SBATCH --job-name=s3rec_ml-1m
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err 
#SBATCH --partition=general 

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH --mem=16G

#SBATCH --gres=gpu:1

#SBATCH --time=9:30:00

#SBATCH --mail-type=END
#SBATCH --mail-user="chunings@andrew.cmu.edu"

# enter a config env
eval "$(conda shell.bash hook)"
conda activate recsys_ben

# Configs
#model="s3rec"
# dataset_type="ml"
dataset="ml-1m"

exp_name="s3rec_ml-1m"

nproc=1

source_dir="/data/user_data/chunings/RecSys-Benchmark/RecBole"

cd $source_dir

python3 run_recbole.py --model "S3Rec" --dataset "ml-1m" --exp_name "s3rec_ml-1m" --nproc 1 --config_files "configs/models/S3Rec.yaml configs/datasets/ml.yaml configs/eval.yaml"