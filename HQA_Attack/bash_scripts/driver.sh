#!/bin/bash
SCRIPT_PATH="run_hqaV4_job.sh"
DATASETS=("imdb" "ag_news" "yelp_polarity" "rotten_tomatoes")
SYNONYM_METHOD="counter-fitted"

for DATASET_NAME in "${DATASETS[@]}"; do
    OUTPUT_FILE="debug/out_${DATASET_NAME}_${SYNONYM_METHOD}.out"
    ERROR_FILE="debug/out_${DATASET_NAME}_${SYNONYM_METHOD}.err"

    echo "Submitting $SCRIPT_PATH for dataset: $DATASET_NAME"

    sbatch -o "$OUTPUT_FILE" -e "$ERROR_FILE" --job-name="${DATASET_NAME}_${SYNONYM_METHOD}" "$SCRIPT_PATH" "$DATASET_NAME" "$SYNONYM_METHOD"
done

echo "All jobs have been submitted to Slurm."

# sbatch -o "debug/out_imdb_counter-fitted_544.out" -e "debug/out_imdb_counter-fitted_544.err" --job-name="imdb" "run_hqaV4_job.sh" "imdb" "counter-fitted" --start_idx 544

# sbatch -o "debug/out_yelp_polarity_counter-fitted_764.out" -e "debug/out_yelp_polarity_counter-fitted_764.err" --job-name="yelp_polarity" "run_hqaV4_job.sh" "yelp_polarity" "counter-fitted" --start_idx 764