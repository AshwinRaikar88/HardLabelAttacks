import os
import argparse
import sys
from datasets import load_dataset

from utils.CSVUtils import export_result_to_csv
from attack_algorithms.HQAAttack import HQAAttack

AVAILABLE_DATASETS = ["imdb", "ag_news", "yelp_polarity", "rotten_tomatoes"]

# HF_TOKEN = os.environ.get('HF_TOKEN')
HF_TOKEN = "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"

model_dict = {
    "imdb": ("text-classification", "textattack/distilbert-base-uncased-imdb", "imdb", "test"),
    "ag_news": ("text-classification", "textattack/distilbert-base-uncased-ag-news", "fancyzhx/ag_news", "test"),
    "yelp_polarity": ("text-classification", "randellcotta/distilbert-base-uncased-finetuned-yelp-polarity", "yelp_polarity", "test"),
    "rotten_tomatoes": ("text-classification", "textattack/distilbert-base-uncased-rotten-tomatoes", "cornell-movie-review-data/rotten_tomatoes", "test")
}


def main(dataset_name):
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    print(f"Starting attack on dataset: {dataset_name}")
    
    attack = HQAAttack(model_name=model_dict[dataset_name][1], hf_token=HF_TOKEN)
    
    dataset = load_dataset(model_dict[dataset_name][2], split="test", token=HF_TOKEN)
    samples = dataset.shuffle(seed=42).select(range(1000))

    results = []
    start_idx = 0

    for idx, sample in enumerate(samples, start=start_idx):
        print(f"\n{'='*70}")
        print(f"Sample {idx + 1}/1000")
        print(f"{'='*70}")

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
        export_result_to_csv(result, filename=f"attack_{dataset_name}.csv")

    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")

    successful = sum(1 for r in results if r['success'])
    total_queries = sum(r['queries'] for r in results)
    avg_queries = total_queries / len(results) if results else 0

    print(f"Total samples: {len(results)}")
    print(f"Successful attacks: {successful}/{len(results)}")
    print(f"Success rate: {100*successful/len(results):.1f}%")
    print(f"Total queries: {total_queries}")
    print(f"Average queries per sample: {avg_queries:.1f}")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Run the LMATK attack on a specified dataset.",
        formatter_class=argparse.RawTextHelpFormatter # Optional: For better help formatting
    )

    parser.add_argument(
        '--dataset', type=str, choices=AVAILABLE_DATASETS, default='imdb',
        help=f"Dataset to run the attack on. "
             f"Available choices:\n{', '.join(AVAILABLE_DATASETS)}\n"
             f"(default: 'imdb')")

    args = parser.parse_args()
    main(args.dataset)
