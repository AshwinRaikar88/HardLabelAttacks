import csv
from datasets import load_dataset

from HQA_Algos import HQA_Algos

from SentimentAnayzer import SentimentAnalyzer



if __name__ == "__main__":
    DEBUG = True
    
    sentiment_analysis = SentimentAnalyzer()
    hqa_attack = HQA_Algos(model=sentiment_analysis.model, vector_file_path='/content/counter-fitted-vectors.txt', DEBUG=DEBUG)
    
    dataset = load_dataset("glue", "sst2")

    split = "validation"
    data_adversarial = {}
    result = []


    for i in range(dataset[split].num_rows):
        if i == 5:            
            break
        
        hqa_attack.reset_params()

        if not DEBUG:
            print(f"Current iteration = {i}")

        orig_sentence = dataset[split][i]["sentence"]
        orig_label = dataset[split][i]["label"]

        if orig_label == 0:
            orig_label = "NEGATIVE"
        else:
            orig_label = "POSITIVE"

        pred_orig_label = sentiment_analysis.get_pred_labels(orig_sentence)        
        
        if orig_label != pred_orig_label:
            print("Skipping due to wrong model predictions")
            continue
        
        random_adv = hqa_attack.generate_random_adversarial_example(orig_sentence, orig_label)

        sub_adv = hqa_attack.substitute_original_words(x=orig_sentence, x_t=random_adv, orig_label=orig_label)

        data_adversarial[i] = (orig_sentence, random_adv, sub_adv)


        if DEBUG:
            print("---"*15)
            print(f"Label: {orig_label}")        
            print(f"Predicted Label: {pred_orig_label}") 
            pred_sub_label = sentiment_analysis.get_pred_labels(sub_adv)
            print(f"Predictions Attack Label: {pred_sub_label}")

            print(f"Sentence (Original): {orig_sentence}")
            print(f"Sentence (Random): {random_adv}")
            print(f"Sentence (Substituted): {sub_adv}")

            print(data_adversarial[i])   
            print("---"*15)       


    # export_data(data = data_adversarial, csv_filename = "evaluation_data.csv")