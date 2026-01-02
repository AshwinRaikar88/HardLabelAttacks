#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH --qos=normal
#SBATCH -o logs/hqa_mistral.out
#SBATCH -e logs/hqa_mistral.err
#SBATCH --nodes=1
#SBATCH --partition=a100-80gb
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=raikaa01@pfw.edu
#SBATCH --mem=40G

# Date  created Dec 3 2025
source /scratch/gilbreth/raikaa01/Projects/Mistral/venv/bin/activate

cd "/scratch/gilbreth/raikaa01/Projects/HardLabelAttacks/HQA_Attack"

python3 scripts/main_mistral_attack.py \
    --model_path "/scratch/gilbreth/raikaa01/Projects/UnslothFinetuning/Mistral/mistral_sentiment_rt_10/final_model" \
    --base_model unsloth/mistral-7b-instruct-v0.3 \
    --synonym_method counter-fitted \
    --embedding_path "/scratch/gilbreth/raikaa01/Projects/HardLabelAttacks/HQA_Attack/weights/counter-fitted-vectors.txt" \
    --num_samples 1000 \
    --max_iterations 3 \
    --hf_token_file /scratch/gilbreth/raikaa01/Projects/HardLabelAttacks/hf_token.txt
