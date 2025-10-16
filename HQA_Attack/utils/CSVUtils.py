import os
import csv

def export_result_to_csv(result, filename="attack_results.csv"):
    """
    Appends a single attack result (dictionary) to a CSV file.
    
    If the file does not exist, it creates the file and writes the header row 
    based on the keys in the result dictionary.
    
    Args:
        result (dict): A dictionary containing the attack result.
        filename (str): The name of the CSV file to write to.
    """
    # 1. Define the field names (header) based on the expected format
    fieldnames = list(result.keys())
    
    # Check if the file already exists
    file_exists = os.path.exists(filename)
    
    # Open the file in append mode ('a'). 
    # 'newline=""' is crucial for consistent CSV writing across platforms.
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            # Create a DictWriter object
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 2. Write the header only if the file is being created
            if not file_exists:
                print(f"Creating new file: {filename} and writing header.")
                writer.writeheader()
                
            # 3. Append the data row
            writer.writerow(result)
            
    except Exception as e:
        print(f"An error occurred while writing to CSV: {e}")