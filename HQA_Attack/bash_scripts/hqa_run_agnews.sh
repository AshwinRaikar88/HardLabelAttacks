#!/bin/bash
#SBATCH -A standby
#SBATCH -o debug/hqa_attack_agnews.out
#SBATCH -e debug/hqa_attack_agnews.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raikara@purdue.edu
#SBATCH --mem=80G

# Activate your virtual environment
source /scratch/gilbreth/raikara/NLP_Research/venv/bin/activate

# pip3 install -r requirements.txt


cd "/home/raikara/NLP Research"

# Run the training script
python3 hqa_attack_agnews.py --set "$1"
