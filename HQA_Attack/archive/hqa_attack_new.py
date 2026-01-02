import csv
import argparse
from tqdm import tqdm
from attack_algorithms.HQA import HQA_Attack
from models.SentimentAnalyzer import SentimentAnalyzer


def export_data(data, max_query_count=5000, csv_filename = "evaluation_data.csv"):
    """
    Export the data to a CSV file.

    data: dictionary containing the data to be exported
    csv_filename: name of the CSV file
    """
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["ID", "Original Sentence", "Generated Sentence", "Substituted Sentence","Query Count", "Attack Success"])

        for key, (original, generated, substituted, query_count, success) in data.items():
            if (query_count == 0 | query_count == max_query_count):
                success = False
            writer.writerow([key, original, generated, substituted, query_count, success])

    print(f"\nCSV file '{csv_filename}' saved successfully!")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ag_news", help="dataset name")
    args = parser.parse_args()   
    
    model_dict = {
        "imdb": ("text-classification", "textattack/distilbert-base-uncased-imdb", "imdb", "test"),
        "ag_news": ("text-classification", "textattack/distilbert-base-uncased-ag-news", "fancyzhx/ag_news", "test"),
        "yelp_polarity": ("text-classification", "randellcotta/distilbert-base-uncased-finetuned-yelp-polarity", "yelp_polarity", "test"),
        "rotten_tomatoes": ("text-classification", "textattack/distilbert-base-uncased-rotten-tomatoes", "cornell-movie-review-data/rotten_tomatoes", "test")
    }

    dataset_name = args.dataset_name
    sentiment_analysis = SentimentAnalyzer(model_dict, data=dataset_name)
    
    hqa_attack = HQA_Attack(model=sentiment_analysis.model)    
    
    data_adversarial = {}
    result = []
    
         

    for i in tqdm(range(1000), desc="Evaluating", ncols=100):                
        orig_sentence = sentiment_analysis.dataset_sample[i]["text"]
        orig_label = sentiment_analysis.dataset_sample[i]["label"]  

        if dataset_name == "ag_news":            
            if orig_label == 0:
                orig_label = "LABEL_0"
            elif orig_label == 1:
                orig_label = "LABEL_1"
            elif orig_label == 2:
                orig_label = "LABEL_2"
            elif orig_label == 3:
                orig_label = "LABEL_3"                
        else:
            if orig_label == 0:
                orig_label = "LABEL_0"
            elif orig_label == 1:
                orig_label = "LABEL_1"

        pred_orig_label = sentiment_analysis.get_pred_labels(orig_sentence)

        if orig_label != pred_orig_label:            
            data_adversarial[i] = (orig_sentence, "---", "---", hqa_attack.get_query_count(), False)
            continue

        random_adv = hqa_attack.generate_random_adversarial_example(orig_sentence, orig_label)

        sub_adv = hqa_attack.substitute_original_words(x=orig_sentence, x_t=random_adv, orig_label=orig_label)

        data_adversarial[i] = (orig_sentence, random_adv, sub_adv, hqa_attack.get_query_count(), True)

        hqa_attack.reset_params()

    export_data(data = data_adversarial, csv_filename = f"./output/evaluation_{dataset_name}.csv")
    print("Attack Done")