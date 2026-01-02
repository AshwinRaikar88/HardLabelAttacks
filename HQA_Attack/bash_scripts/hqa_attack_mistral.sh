#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH --qos=normal
#SBATCH --job-name=HQA_mistral_imdb
#SBATCH -o logs/attack/mistral/imdb.out
#SBATCH -e logs/attack/mistral/imdb.err
#SBATCH --nodes=1
#SBATCH --partition=a100-80gb
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=raikaa01@pfw.edu
#SBATCH --mem=40G

source /scratch/gilbreth/raikaa01/Projects/UnslothFinetuning/venv/bin/activate

cd "/scratch/gilbreth/raikaa01/Projects/HardLabelAttacks/HQA_Attack"

# Rotten Tomatoes with counter-fitted embeddings
# python hqa_mistral.py \
#     --model_path "/scratch/gilbreth/raikaa01/Projects/UnslothFinetuning/Mistral/mistral_rt/final_model" \
#     --dataset rotten_tomatoes \
#     --synonym_method counter-fitted \
#     --embedding_path "/scratch/gilbreth/raikaa01/Projects/HardLabelAttacks/HQA_Attack/weights/counter-fitted-vectors.txt" \
#     --num_samples 1000 \
#     --max_iterations 3 \
#     --hf_token "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" \
#     --output_file output/llm_attacks/mistral_rt_logits_attack.json

# IMDB with counter-fitted embeddings for Mistral
python hqa_mistral.py \
    --model_path "/scratch/gilbreth/raikaa01/Projects/UnslothFinetuning/Mistral/mistral_imdb/final_model" \
    --dataset imdb \
    --synonym_method counter-fitted \
    --embedding_path "/scratch/gilbreth/raikaa01/Projects/HardLabelAttacks/HQA_Attack/weights/counter-fitted-vectors.txt" \
    --num_samples 1000 \
    --max_iterations 3 \
    --start_idx 300 \
    --end_idx 350 \
    --hf_token "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" \
    --output_file output/llm_attacks/mistral_imdb_logits_attack.json \
    --checkpoint_file output/llm_attacks/mistral_imdb_logits_attack.checkpoint.json \
    --checkpoint_interval 5 \
    --resume