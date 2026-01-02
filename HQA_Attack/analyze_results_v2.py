from utils.CSVUtils import analyze

if __name__ == "__main__":
    print("Analyzing output files")
    csv_path = f"/scratch/gilbreth/raikaa01/Projects/HardLabelAttacks/HQA_Attack/output/mistral_counter-fitted/attack_mistral_counter-fitted_20251204_132740.csv" 
    outfile = f"output/reports/report_mistral_counter-fitted.csv"
    analyze(csv_path, outfile)