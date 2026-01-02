import numpy as np
import os
import json
from datetime import datetime, timedelta

def load_checkpoint(checkpoint_file):
    """Load existing checkpoint if available"""
    if os.path.exists(checkpoint_file):
        print(f"\n{'=' * 80}")
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_file}")
        print(f"{'=' * 80}\n")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint
    return None


def save_checkpoint(checkpoint_file, results, start_idx, total_samples,
                    success_count, failed_count, skipped_count, dataset,
                    total_time, sample_times, batch_start=None, batch_end=None):
    """Save checkpoint with current progress"""
    valid_attempts = (start_idx + 1) - skipped_count

    checkpoint = {
        'dataset': dataset,
        'method': 'logits',
        'last_completed_index': start_idx,
        'total_samples': total_samples,
        'successful_attacks': success_count,
        'failed_attacks': failed_count,
        'skipped_samples': skipped_count,
        'valid_attempts': valid_attempts,
        'success_rate': success_count / valid_attempts if valid_attempts > 0 else 0,
        'total_time_seconds': total_time,
        'avg_time_per_attack': np.mean(sample_times) if sample_times else 0,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

    # Add batch info if provided
    if batch_start is not None:
        checkpoint['batch_start_idx'] = batch_start
    if batch_end is not None:
        checkpoint['batch_end_idx'] = batch_end

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\n[CHECKPOINT SAVED] Progress: {start_idx + 1}/{total_samples} samples")


def format_time(seconds):
    """Format seconds into human-readable time"""
    return str(timedelta(seconds=int(seconds)))