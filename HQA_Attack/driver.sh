#!/bin/bash
SCRIPT_PATH="run_hqaV4_job.sh"
DATASETS=("imdb" "ag_news" "yelp_polarity" "rotten_tomatoes")
SYNONYM_METHOD="counter-fitted"

for DATASET_NAME in "${DATASETS[@]}"; do
    OUTPUT_FILE="debug/out_${DATASET_NAME}_${SYNONYM_METHOD}.out"
    ERROR_FILE="debug/out_${DATASET_NAME}_${SYNONYM_METHOD}.err"

    echo "Submitting $SCRIPT_PATH for dataset: $DATASET_NAME"

    sbatch -o "$OUTPUT_FILE" -e "$ERROR_FILE" "$SCRIPT_PATH" "$DATASET_NAME" "$SYNONYM_METHOD"
done

echo "All jobs have been submitted to Slurm."