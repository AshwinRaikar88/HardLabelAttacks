import csv
import argparse
from tqdm import tqdm
from attack_algorithms.HQA import HQA_Attack
from models.SentimentAnalyzer_AGNews import SentimentAnalyzer_AG_News


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
    sentiment_analysis = SentimentAnalyzer_AG_News()
    hqa_attack = HQA_Attack(model=sentiment_analysis.model)    
    
    data_adversarial = {}
    result = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=int, default=1, help="Set number to export")
    args = parser.parse_args()
    
    set_num = args.set 
    if set_num == 1:
        start = 0
        end = 1001        
    # elif set_num == 2:
    #     start = 1001
    #     end = 2001
    # elif set_num == 3:
    #     start = 2001
    #     end = 3001
    # elif set_num == 4:
    #     start = 3001
    #     end = 4001
    # elif set_num == 5:
    #     start = 4001
    #     end = 5001
    # elif set_num == 6:
    #     start = 5001
    #     end = 6001
    # elif set_num == 7:
    #     start = 6001
    #     end = 7001
    # elif set_num == 8:
    #     start = 7001
    #     end = 7600    

    for i in tqdm(range(start, end), desc="Evaluating", ncols=100):                
        orig_sentence = sentiment_analysis.dataset_sample[i]["text"]
        orig_label = sentiment_analysis.dataset_sample[i]["label"]  

        if orig_label == 0:
            orig_label = "LABEL_0"
        elif orig_label == 1:
            orig_label = "LABEL_1"
        elif orig_label == 2:
            orig_label = "LABEL_2"
        elif orig_label == 3:
            orig_label = "LABEL_3"        

        pred_orig_label = sentiment_analysis.get_pred_labels(orig_sentence)

        if orig_label != pred_orig_label:            
            data_adversarial[i] = (orig_sentence, "---", "---", hqa_attack.get_query_count(), False)
            continue

        random_adv = hqa_attack.generate_random_adversarial_example(orig_sentence, orig_label)

        sub_adv = hqa_attack.substitute_original_words(x=orig_sentence, x_t=random_adv, orig_label=orig_label)

        data_adversarial[i] = (orig_sentence, random_adv, sub_adv, hqa_attack.get_query_count(), True)

        hqa_attack.reset_params()

    export_data(data = data_adversarial, csv_filename = f"./output/evaluation_ag_news_data_set.csv")
    print("Attack Done")