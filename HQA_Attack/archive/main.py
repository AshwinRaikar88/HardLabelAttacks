import csv
from datasets import load_dataset
from tqdm import tqdm
from attack_algorithms.HQA import HQA_Attack
from models.SentimentAnalyzer import SentimentAnalyzer

def export_data(data, csv_filename = "evaluation_data.csv"):
  """
  Export the data to a CSV file.

  data: dictionary containing the data to be exported
  csv_filename: name of the CSV file
  """
  with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
      writer = csv.writer(file)

      writer.writerow(["ID", "Original Sentence", "Generated Sentence", "Substituted Sentence","Query Count", "Attack Success"])

      for key, (original, generated, substituted, query_count, success) in data.items():
          writer.writerow([key, original, generated, substituted, query_count, success])

  print(f"\nCSV file '{csv_filename}' saved successfully!")




if __name__ == "__main__":
    DEBUG = False

    sentiment_analysis = SentimentAnalyzer()
    hqa_attack = HQA_Attack(model=sentiment_analysis.model)

    dataset = load_dataset("glue", "sst2")

    split = "validation"
    data_adversarial = {}
    result = []

    for i in tqdm(range(dataset[split].num_rows), desc="Evaluating", ncols=100):

        if DEBUG:
            print(f"Current iteration = {i}")

        orig_sentence = dataset[split][i]["sentence"]
        orig_label = dataset[split][i]["label"]

        orig_label = "NEGATIVE" if orig_label == 0 else "POSITIVE"

        pred_orig_label = sentiment_analysis.get_pred_labels(orig_sentence)

        if orig_label != pred_orig_label:
            if DEBUG:
                print("Skipping due to wrong model predictions")
            data_adversarial[i] = (orig_sentence, "---", "---", hqa_attack.get_query_count(), False)
            continue

        random_adv = hqa_attack.generate_random_adversarial_example(orig_sentence, orig_label)

        sub_adv = hqa_attack.substitute_original_words(x=orig_sentence, x_t=random_adv, orig_label=orig_label)

        data_adversarial[i] = (orig_sentence, random_adv, sub_adv, hqa_attack.get_query_count(), True)

        hqa_attack.reset_params()

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


    export_data(data = data_adversarial, csv_filename = "evaluation_data.csv")
    print("Attack Done")