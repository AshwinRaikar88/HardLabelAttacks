from utils.CSVUtils import analyze

if __name__ == "__main__":
    print("Analyzing output files")
    DATASET = ("imdb", "ag_news", "yelp_polarity", "rotten_tomatoes")
    SYNONYM_METHOD = "counter-fitted"

    for dataset in DATASET:
        csv_path = f"output/{SYNONYM_METHOD}/attack_{dataset}_{SYNONYM_METHOD}.csv" 
        outfile = f"output/reports/report_{dataset}_{SYNONYM_METHOD}.csv"

        print(f"Dataset: {dataset}")
        print(f"Synonym method: {SYNONYM_METHOD}")
        analyze(csv_path, outfile)