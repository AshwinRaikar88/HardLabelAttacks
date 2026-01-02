import nltk
import warnings
import argparse
import time
from datasets import load_dataset
from utils.attack_utils import *
from hqa_attack import HQAAttack

warnings.filterwarnings('ignore')

os.environ["HF_HOME"] = "/scratch/gilbreth/raikaa01/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/gilbreth/raikaa01/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/gilbreth/raikaa01/hf_cache/datasets"
os.environ["TORCH_HOME"] = "/scratch/gilbreth/raikaa01/torch_cache"

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HQA Attack")
    parser.add_argument("--llm_model", type=str, default="mistral")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--dataset", type=str, default="rotten_tomatoes",
                        choices=['rotten_tomatoes', 'imdb', 'ag_news'])
    parser.add_argument("--synonym_method", type=str, default="wordnet",
                        choices=['wordnet', 'counter-fitted', 'glove'])
    parser.add_argument("--embedding_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index for batch processing (0-indexed, inclusive)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for batch processing (inclusive)")
    parser.add_argument("--output_file", type=str, default="hqa_attack_results_logits.json")
    parser.add_argument("--checkpoint_file", type=str, default=None,
                        help="Checkpoint file for resume (default: <output_file>.checkpoint)")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Save checkpoint every N samples (default: 10)")
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--resume", action='store_true',
                        help="Resume from checkpoint if available")

    args = parser.parse_args()

    model_dict = {
        "imdb": ("text-classification", "textattack/distilbert-base-uncased-imdb", "imdb", "test"),
        "ag_news": ("text-classification", "textattack/distilbert-base-uncased-ag-news", "fancyzhx/ag_news", "test"),
        "yelp_polarity": ("text-classification", "randellcotta/distilbert-base-uncased-finetuned-yelp-polarity",
                          "yelp_polarity", "test"),
        "rotten_tomatoes": ("text-classification", "textattack/distilbert-base-uncased-rotten-tomatoes",
                            "cornell-movie-review-data/rotten_tomatoes", "test")
    }

    print(f"\nLoading {args.dataset} dataset...")
    dataset = load_dataset(model_dict[args.dataset][2], split="test", token=args.hf_token)
    full_test_data = dataset.shuffle(seed=42).select(range(args.num_samples))

    # Handle batch range
    batch_start = args.start_idx if args.start_idx is not None else 0
    batch_end = args.end_idx

    # Step 1: First select the subset based on num_samples (if provided)
    if args.num_samples and args.num_samples < len(full_test_data):
        working_dataset = full_test_data.select(range(args.num_samples))
        total_available = args.num_samples
        print(f"Working with first {args.num_samples} samples from test set")
    else:
        working_dataset = full_test_data
        total_available = len(full_test_data)
        print(f"Working with full test set ({total_available} samples)")

    # Step 2: Determine batch range within the working dataset
    if batch_end is None:
        batch_end = total_available - 1

    # Validate batch range
    if batch_start < 0:
        raise ValueError(f"start_idx must be >= 0, got {batch_start}")
    if batch_end >= total_available:
        raise ValueError(f"end_idx ({batch_end}) must be < {total_available} (available samples)")
    if batch_start > batch_end:
        raise ValueError(f"start_idx ({batch_start}) must be <= end_idx ({batch_end})")

    # Step 3: Select the batch slice from working dataset
    test_data = working_dataset.select(range(batch_start, batch_end + 1))

    print(f"Batch range: [{batch_start}, {batch_end}] (inclusive)")
    total_samples = len(test_data)
    print(f"Processing {total_samples} samples from this batch")

    # Set checkpoint and output files with batch suffix
    if args.start_idx is not None or args.end_idx is not None:
        start_suffix = args.start_idx if args.start_idx is not None else 0
        end_suffix = args.end_idx if args.end_idx is not None else "end"
        batch_suffix = f"_{start_suffix}_to_{end_suffix}"

        base_name = args.output_file.replace('.json', '')
        args.output_file = f"{base_name}{batch_suffix}.json"

        if args.checkpoint_file is None:
            args.checkpoint_file = f"{base_name}{batch_suffix}.checkpoint.json"
    else:
        if args.checkpoint_file is None:
            args.checkpoint_file = args.output_file.replace('.json', '.checkpoint.json')

    # Try to load checkpoint
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(args.checkpoint_file)

    # Initialize timing and counters
    total_start_time = time.time()
    sample_times = []

    if checkpoint:
        results = checkpoint['results']
        start_idx = checkpoint['last_completed_index'] + 1
        success_count = checkpoint['successful_attacks']
        failed_count = checkpoint.get('failed_attacks', 0)
        skipped_count = checkpoint.get('skipped_samples', 0)
        previous_time = checkpoint.get('total_time_seconds', 0)
        sample_times = [r.get('attack_time', 0) for r in results if 'attack_time' in r]
        print(f"Resuming from sample {start_idx}/{checkpoint['total_samples']}")
        print(f"Previous progress: {success_count} successful, {failed_count} failed, {skipped_count} skipped")
        print(f"Previous runtime: {format_time(previous_time)}")
    else:
        results = []
        start_idx = 0
        success_count = 0
        failed_count = 0
        skipped_count = 0
        previous_time = 0

    if args.dataset == 'imdb':
        max_seq_len = 2048
    else:
        max_seq_len = 512

    # Initialize attack
    attacker = HQAAttack(
        llm_model=args.llm_model,
        model_path=args.model_path,
        dataset=args.dataset,
        synonym_method=args.synonym_method,
        embedding_path=args.embedding_path,
        max_seq_length=max_seq_len,
        hf_token=args.hf_token
    )

    # Run attacks
    for i in range(total_samples):
        current_idx = batch_start + i
        sample_start_time = time.time()

        print(f"\n{'=' * 80}")
        print(f"Sample {current_idx}/{batch_start + total_samples - 1}")
        elapsed_time = time.time() - total_start_time + previous_time
        avg_time = np.mean(sample_times) if sample_times else 0
        remaining = (len(test_data) - i) * avg_time
        print(f"Elapsed: {format_time(elapsed_time)} | "
              f"Avg/sample: {avg_time:.1f}s | "
              f"ETA: {format_time(remaining)}")
        print(f"{'=' * 80}")

        example = test_data[i]
        text = example['text']
        true_label = example['label']  # Get ground truth

        try:
            result = attacker.attack(
                text,
                true_label=true_label,  # Pass ground truth
                max_iterations=args.max_iterations,
                verbose=True
            )
            results.append(result)

            # Update counters based on result
            if result.get('skipped', False):
                skipped_count += 1
            elif result['success']:
                success_count += 1
            else:
                failed_count += 1

            sample_time = time.time() - sample_start_time
            sample_times.append(sample_time)

            # Print sample summary
            print(f"\n{'─' * 80}")
            print(f"Sample {i + 1} completed in {sample_time:.2f}s")
            if result.get('skipped', False):
                print(f"Status: SKIPPED (original misclassification)")
            else:
                print(f"Success: {result['success']} | Queries: {result['queries']}")

            valid_attempts = (i + 1) - skipped_count
            if valid_attempts > 0:
                attack_success_rate = success_count / valid_attempts * 100
                print(f"Attack success rate: {success_count}/{valid_attempts} ({attack_success_rate:.2f}%)")
            print(f"Overall stats: {success_count} success, {failed_count} failed, {skipped_count} skipped")
            print(f"{'─' * 80}")

            # Save checkpoint at intervals
            if (i + 1) % args.checkpoint_interval == 0 or (i + 1) == len(test_data):
                current_time = time.time() - total_start_time + previous_time
                save_checkpoint(
                    args.checkpoint_file,
                    results,
                    current_idx,
                    len(test_data),
                    success_count,
                    failed_count,
                    skipped_count,
                    args.dataset,
                    current_time,
                    sample_times
                )

        except Exception as e:
            print(f"\n[ERROR] Failed on sample {i + 1}: {str(e)}")
            # Save checkpoint on error
            current_time = time.time() - total_start_time + previous_time
            save_checkpoint(
                args.checkpoint_file,
                results,
                current_idx - 1,
                len(test_data),
                success_count,
                failed_count,
                skipped_count,
                args.dataset,
                current_time,
                sample_times
            )
            raise

    # Calculate final statistics
    total_time = time.time() - total_start_time + previous_time
    avg_time_per_attack = np.mean(sample_times) if sample_times else 0
    valid_attempts = len(test_data) - skipped_count

    # Save final results
    summary = {
        'dataset': args.dataset,
        'method': 'logits',
        'total_samples': len(test_data),
        'successful_attacks': success_count,
        'failed_attacks': failed_count,
        'skipped_samples': skipped_count,
        'valid_attempts': valid_attempts,
        'success_rate': success_count / valid_attempts if valid_attempts > 0 else 0,
        'total_time_seconds': total_time,
        'total_time_formatted': format_time(total_time),
        'avg_time_per_attack': avg_time_per_attack,
        'min_time_per_attack': min(sample_times) if sample_times else 0,
        'max_time_per_attack': max(sample_times) if sample_times else 0,
        'results': results,
        'completed_at': datetime.now().isoformat()
    }

    with open(args.output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"ATTACK ON {args.dataset} COMPLETED")
    print(f"{'=' * 80}")
    print(f"Dataset: {args.dataset}")
    print(f"Method: Logits")
    print(f"Total samples processed: {len(test_data)}")
    print(f"{'=' * 80}")
    print(f"ATTACK RESULTS")
    print(f"{'=' * 80}")
    print(f"Successful attacks: {success_count}")
    print(f"Failed attacks: {failed_count}")
    print(f"Skipped (originally misclassified): {skipped_count}")
    print(f"Valid attack attempts: {valid_attempts}")
    if valid_attempts > 0:
        print(f"Attack success rate: {success_count / valid_attempts * 100:.2f}% ({success_count}/{valid_attempts})")
    print(f"{'=' * 80}")
    print(f"TIMING STATISTICS")
    print(f"{'=' * 80}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Average time per sample: {avg_time_per_attack:.2f}s")
    print(f"Fastest attack: {min(sample_times) if sample_times else 0:.2f}s")
    print(f"Slowest attack: {max(sample_times) if sample_times else 0:.2f}s")
    print(f"{'=' * 80}")
    print(f"Results saved to: {args.output_file}")

    # Clean up checkpoint file after successful completion
    if os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)
        print(f"Checkpoint file removed: {args.checkpoint_file}")