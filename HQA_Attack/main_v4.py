import os
import argparse
import sys
from datasets import load_dataset

from utils.token_loader import read_hf_token
from utils.CSVUtils import export_result_to_csv
from attack_algorithms.hqa_attack_v4 import HQAAttack

AVAILABLE_DATASETS = ["imdb", "ag_news", "yelp_polarity", "rotten_tomatoes"]

# HF_TOKEN = os.environ.get('HF_TOKEN')
HF_TOKEN = read_hf_token("hf_token.txt")

model_dict = {
    "imdb": ("text-classification", "textattack/distilbert-base-uncased-imdb", "imdb", "test"),
    "ag_news": ("text-classification", "textattack/distilbert-base-uncased-ag-news", "fancyzhx/ag_news", "test"),
    "yelp_polarity": ("text-classification", "randellcotta/distilbert-base-uncased-finetuned-yelp-polarity", "yelp_polarity", "test"),
    "rotten_tomatoes": ("text-classification", "textattack/distilbert-base-uncased-rotten-tomatoes", "cornell-movie-review-data/rotten_tomatoes", "test")
}


def main(dataset_name, synonym_method):
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    if synonym_method == 'counter-fitted':
        embedding_path = "/scratch/gilbreth/raikaa01/Projects/NLP Research/weights/counter-fitted-vectors.txt"
    elif synonym_method == 'glove':
        embedding_path = "/scratch/gilbreth/jrusert/glove.840B.300d.txt"
    else:
        embedding_path = None

    print(f"Starting attack on dataset: {dataset_name}")
      
    dataset = load_dataset(model_dict[dataset_name][2], split="test", token=HF_TOKEN)
    samples = dataset.shuffle(seed=42).select(range(1000))

    # Create label map from dataset info
    label_map = {}
    if hasattr(dataset, 'features') and 'label' in dataset.features:
        label_feature = dataset.features['label']
        if hasattr(label_feature, 'names'):
            label_map = {i: name for i, name in enumerate(label_feature.names)}
    
    # Fallback to default if no label info available
    if not label_map:
        label_map = {0: "Label_0", 1: "Label_1", 2: "Label_2", 3: "Label_3"}
    
    print(f"Label mapping: {label_map}")

    print("\n" + "="*70)
    print("HQA-Attack: Running on 1000 AG News Samples")
    print(f"Using synonym method: {synonym_method}")
    print("="*70)
    
    attack = HQAAttack(model_name=model_dict[dataset_name][1], 
                       synonym_method=synonym_method,
                       embedding_path=embedding_path,
                       label_map=label_map, 
                       hf_token=HF_TOKEN)

    results = []
    start_idx = 0

    for idx, sample in enumerate(samples, start=start_idx):
        print(f"\n{'='*70}")
        print(f"Sample {idx + 1}/1000")
        print(f"{'='*70}")

        # Check if the sample is already adversarial
        original_label = attack.label_map[sample['label']]
        predicted_label = attack.get_prediction(sample['text'])[0]
        predicted_label = attack.label_map[predicted_label]

        # print(f"Original label: {original_label}")
        # print(f"Predicted label: {predicted_label}")

        if original_label != predicted_label:
          print("Skipping")
          result = {'original': sample['text'],
                    'adversarial': "N/A",
                    'original_label': original_label,
                    'final_label': predicted_label,
                    'success': "Skipped",
                    'queries': 0}
        else:
          result = attack.attack(sample['text'], max_iterations=3, verbose=True)

        results.append(result)
        export_result_to_csv(result, filename=f"output/attack_{dataset_name}_{synonym_method}.csv")

    print(f"\n{'='*70}")
    print("Attack completed")
    print(f"{'='*70}")



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Run the LMATK attack on a specified dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--dataset', type=str, choices=AVAILABLE_DATASETS, default='imdb',
        help=f"Dataset to run the attack on. "
             f"Available choices:\n{', '.join(AVAILABLE_DATASETS)}\n"
             f"(default: 'imdb')")

    parser.add_argument(
        '--synonym_method', type=str, choices=['wordnet', 'counter-fitted', 'glove'], default='wordnet',
        help=f"Synonym method to use. "
             f"Available choices:\n{', '.join(['wordnet', 'counter-fitted', 'glove'])}\n"
             f"(default: 'wordnet')")

    args = parser.parse_args()
    main(args.dataset, args.synonym_method)
