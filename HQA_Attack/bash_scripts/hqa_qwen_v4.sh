DATASET="imdb"
MODEL_PATH="/scratch/gilbreth/raikaa01/Projects/UnslothFinetuning/Qwen/qwen3-8B_imdb_output/final_model/"
SYNONYM_METHOD="counter-fitted"
EMBEDDING_PATH="/scratch/gilbreth/raikaa01/Projects/HardLabelAttacks/HQA_Attack/weights/counter-fitted-vectors.txt"
NUM_SAMPLES=1000
MAX_ITERATIONS=3
START_IDX=951
END_IDX=999
HF_TOKEN="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"
OUTPUT_FILE="output/llm_attacks/qwen3_${DATASET}_logits_attack_${START_IDX}_${END_IDX}.json"
CHECKPOINT_FILE="output/llm_attacks/checkpoints/qwen3_${DATASET}_logits_attack_${START_IDX}_${END_IDX}.checkpoint.json"
CHECKPOINT_INTERVAL=5

sbatch <<EOT
#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH --qos=standby
#SBATCH --job-name=${START_IDX}_${END_IDX}_${DATASET}_HQA_qwen3
#SBATCH -o logs/attack/qwen3/${DATASET}_${START_IDX}_${END_IDX}.out
#SBATCH -e logs/attack/qwen3/${DATASET}_${START_IDX}_${END_IDX}.err
#SBATCH --nodes=1
#SBATCH --partition=a100-80gb
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=raikaa01@pfw.edu
#SBATCH --mem=40G

source /scratch/gilbreth/raikaa01/Projects/UnslothFinetuning/venv/bin/activate

cd "/scratch/gilbreth/raikaa01/Projects/HardLabelAttacks/HQA_Attack"

# IMDB with counter-fitted embeddings for Qwen3
python hqa_qwen3.py \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --synonym_method "$SYNONYM_METHOD" \
    --embedding_path "$EMBEDDING_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --max_iterations "$MAX_ITERATIONS" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --hf_token "$HF_TOKEN" \
    --output_file "$OUTPUT_FILE" \
    --checkpoint_file "$CHECKPOINT_FILE" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL"
# --resume
EOT