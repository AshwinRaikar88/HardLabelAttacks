import os
import argparse
import sys
from datasets import load_dataset
from attack_algorithms.hqa_attack_mistral import HQAAttackMistral
import csv
from datetime import datetime

def read_hf_token(filepath):
    """Read HuggingFace token from file"""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def export_result_to_csv(result, filename="attack_results.csv"):
    """Export a single attack result to CSV file"""
    file_exists = os.path.isfile(filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['original', 'adversarial', 'original_label', 'final_label', 'success', 'queries']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(result)


def main():
    parser = argparse.ArgumentParser(
        description="Run HQA-Attack on fine-tuned Mistral model for Rotten Tomatoes dataset",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to fine-tuned Mistral model (LoRA adapters or full model)"
    )
    
    parser.add_argument(
        '--base_model',
        type=str,
        default="unsloth/mistral-7b-instruct-v0.3",
        help="Base model name (default: unsloth/mistral-7b-instruct-v0.3)"
    )
    
    parser.add_argument(
        '--synonym_method',
        type=str,
        choices=['wordnet', 'counter-fitted', 'glove'],
        default='wordnet',
        help="Synonym method to use (default: wordnet)"
    )
    
    parser.add_argument(
        '--embedding_path',
        type=str,
        default=None,
        help="Path to embedding file (required for counter-fitted and glove methods)"
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help="Number of samples to attack (default: 1000)"
    )
    
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help="Starting index for processing samples (default: 0)"
    )
    
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=3,
        help="Maximum iterations for optimization (default: 3)"
    )
    
    parser.add_argument(
        '--replacement_pct',
        type=float,
        default=0.8,
        help="Percentage of words to replace during initialization (default: 0.8)"
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help="Output directory for results (default: output)"
    )
    
    parser.add_argument(
        '--hf_token_file',
        type=str,
        default=None,
        help="Path to HuggingFace token file"
    )
    
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Get HuggingFace token
    if args.hf_token_file:
        HF_TOKEN = read_hf_token(args.hf_token_file)
    else:
        HF_TOKEN = os.environ.get('HF_TOKEN')
    
    if not HF_TOKEN:
        print("Warning: No HF_TOKEN found. Some datasets may not be accessible.")
    
    # Validate embedding path for certain methods
    if args.synonym_method in ['counter-fitted', 'glove'] and not args.embedding_path:
        print(f"Error: --embedding_path required for {args.synonym_method} method")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir, 
        f"attack_mistral_{args.synonym_method}_{timestamp}.csv"
    )
    
    print("\n" + "="*80)
    print("HQA-Attack on Fine-tuned Mistral Model")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Base model: {args.base_model}")
    print(f"Synonym method: {args.synonym_method}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Starting index: {args.start_idx}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Replacement percentage: {args.replacement_pct}")
    print(f"Output file: {output_file}")
    print("="*80 + "\n")
    
    # Load Rotten Tomatoes dataset
    print("Loading Rotten Tomatoes dataset...")
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test", token=HF_TOKEN)
    samples = dataset.shuffle(seed=42).select(range(args.num_samples))
    print(f"Loaded {len(samples)} samples\n")
    
    # Initialize HQA attack
    print("Initializing HQA attack for Mistral...")
    attack = HQAAttackMistral(
        model_path=args.model_path,
        base_model=args.base_model,
        synonym_method=args.synonym_method,
        embedding_path=args.embedding_path,
        max_seq_length=args.max_seq_length,
        hf_token=HF_TOKEN
    )
    
    # Run attacks
    results = []
    successful_attacks = 0
    skipped_samples = 0
    failed_attacks = 0
    
    for idx, sample in enumerate(samples.select(range(args.start_idx, len(samples))), start=args.start_idx+1):
        print(f"\n{'='*80}")
        print(f"Sample {idx}/{args.num_samples}")
        print(f"{'='*80}")
        
        # Get original label from dataset
        original_label = sample['label']
        original_label_name = attack.label_map[original_label]
        
        # Get model prediction
        predicted_label, pred_confidence = attack.get_prediction(sample['text'])
        predicted_label_name = attack.label_map[predicted_label]
        
        print(f"True label: {original_label_name}")
        print(f"Predicted label: {predicted_label_name} (confidence: {pred_confidence:.3f})")
        
        # Skip if already misclassified
        if original_label != predicted_label:
            print("Sample is already misclassified - skipping")
            result = {
                'original': sample['text'],
                'adversarial': "N/A",
                'original_label': original_label_name,
                'final_label': predicted_label_name,
                'success': "Skipped",
                'queries': 1
            }
            skipped_samples += 1
        else:
            # Run attack
            result = attack.attack(
                sample['text'], 
                max_iterations=args.max_iterations, 
                verbose=True
            )
            
            # Update result with label names instead of indices
            result['original_label'] = attack.label_map[result['original_label']]
            if result['final_label'] is not None:
                result['final_label'] = attack.label_map[result['final_label']]
            
            if result['success']:
                successful_attacks += 1
            else:
                failed_attacks += 1
        
        results.append(result)
        export_result_to_csv(result, filename=output_file)
        
        # Print summary statistics
        total_attempted = idx - args.start_idx
        if total_attempted > 0:
            success_rate = (successful_attacks / (total_attempted - skipped_samples)) * 100 if (total_attempted - skipped_samples) > 0 else 0
            print(f"\nProgress: {total_attempted}/{args.num_samples - args.start_idx}")
            print(f"Successful attacks: {successful_attacks}")
            print(f"Failed attacks: {failed_attacks}")
            print(f"Skipped: {skipped_samples}")
            print(f"Success rate: {success_rate:.2f}%")
    
    # Final summary
    print(f"\n{'='*80}")
    print("Attack Completed")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Successful attacks: {successful_attacks}")
    print(f"Failed attacks: {failed_attacks}")
    print(f"Skipped samples: {skipped_samples}")
    if (len(results) - skipped_samples) > 0:
        success_rate = (successful_attacks / (len(results) - skipped_samples)) * 100
        print(f"Overall success rate: {success_rate:.2f}%")
    print(f"\nResults saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()