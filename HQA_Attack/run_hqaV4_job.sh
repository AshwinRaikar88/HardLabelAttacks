#!/bin/bash
#SBATCH -A pfw-cs
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --partition=v100
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
##SBATCH --mail-type=ALL
#SBATCH --mail-user=raikaa01@pfw.edu
#SBATCH --mem=30G

DATASET_NAME=$1
SYNONYM_METHOD=$2


# Activate your virtual environment
source /scratch/gilbreth/raikaa01/Projects/Mistral/venv/bin/activate

# pip3 install -r requirements.txt


cd "/scratch/gilbreth/raikaa01/Projects/HardLabelAttacks/HQA_Attack"

# Run the training script
python3 main_v4.py --dataset "${DATASET_NAME}" --synonym_method "${SYNONYM_METHOD}"