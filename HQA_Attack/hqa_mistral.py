import torch
import numpy as np
from datasets import load_dataset
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import warnings
import pickle
from unsloth import FastLanguageModel
from peft import PeftModel
import os
import argparse
import json
import time
from datetime import datetime, timedelta

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


class EmbeddingLoader:
    """Load and manage different embedding sources"""
    
    @staticmethod
    def load_counter_fitted(embedding_path):
        """Load counter-fitted word vectors"""
        print(f"Loading counter-fitted embeddings from {embedding_path}...")
        embeddings = {}
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    embeddings[word] = vector
        print(f"Loaded {len(embeddings)} word vectors")
        return embeddings
    
    @staticmethod
    def load_glove(embedding_path):
        """Load GloVe embeddings"""
        print(f"Loading GloVe embeddings from {embedding_path}...")
        embeddings = {}
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    embeddings[word] = vector
        print(f"Loaded {len(embeddings)} word vectors")
        return embeddings


class SynonymExtractor:
    """Extract synonyms using different methods"""
    
    def __init__(self, method='wordnet', embedding_path=None):
        self.method = method
        self.embeddings = None
        
        if method in ['counter-fitted', 'glove']:
            if not embedding_path:
                raise ValueError(f"embedding_path required for {method} method")
            
            if method == 'counter-fitted':
                self.embeddings = EmbeddingLoader.load_counter_fitted(embedding_path)
            else:
                self.embeddings = EmbeddingLoader.load_glove(embedding_path)
            
            self.word2idx = {word: idx for idx, word in enumerate(self.embeddings.keys())}
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
            
            print("Pre-computing cosine similarity matrix...")
            self.sim_matrix = self._compute_similarity_matrix()
            print("Similarity matrix ready")
        
        print(f"Synonym extractor initialized with method: {method}")
    
    def _compute_similarity_matrix(self):
        embeddings_array = np.array([self.embeddings[word] for word in sorted(self.embeddings.keys())])
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)
        sim_matrix = np.dot(embeddings_array, embeddings_array.T)
        return sim_matrix
    
    def get_synonyms(self, word, top_k=50):
        if self.method == 'wordnet':
            return self._get_wordnet_synonyms(word, top_k)
        elif self.method in ['counter-fitted', 'glove']:
            return self._get_embedding_synonyms(word, top_k)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _get_wordnet_synonyms(self, word, top_k=50):
        synonyms = set()
        for synset in wordnet.synsets(word.lower()):
            for lemma in synset.lemmas():
                similar_word = lemma.name().replace('_', ' ')
                if similar_word.lower() != word.lower():
                    synonyms.add(similar_word)
        return list(synonyms)[:top_k]
    
    def _get_embedding_synonyms(self, word, top_k=50):
        word_lower = word.lower()
        if word_lower not in self.word2idx:
            return []
        
        word_idx = self.word2idx[word_lower]
        similarities = self.sim_matrix[word_idx]
        top_indices = np.argsort(-similarities)[1:top_k+1]
        synonyms = [self.idx2word[idx] for idx in top_indices if similarities[idx] > 0.5]
        return synonyms


class MistralClassifierLogits:
    """Mistral classifier using logits-based prediction"""
    
    def __init__(self, model_path, base_model="unsloth/mistral-7b-instruct-v0.3", 
                 max_seq_length=512, hf_token=None, device="cuda", dataset="rotten_tomatoes"):
        self.device = device
        self.max_seq_length = max_seq_length
        self.dataset = dataset
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'rotten_tomatoes': {
                'labels': ['negative', 'positive'],
                'num_classes': 2,
                'task_type': 'sentiment'
            },
            'imdb': {
                'labels': ['negative', 'positive'],
                'num_classes': 2,
                'task_type': 'sentiment'
            },
            'ag_news': {
                'labels': ['world', 'sports', 'business', 'sci/tech'],
                'num_classes': 4,
                'task_type': 'topic'
            }
        }
        
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        config = self.dataset_configs[dataset]
        self.labels = config['labels']
        self.num_classes = config['num_classes']
        self.task_type = config['task_type']
        
        # Label mapping (lowercase for token matching)
        self.label_map = {i: label.lower() for i, label in enumerate(self.labels)}
        self.reverse_label_map = {label.lower(): i for i, label in enumerate(self.labels)}
        
        # Load model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_adapter = os.path.exists(adapter_config_path)
        
        if is_adapter:
            print(f"Loading base model: {base_model}")
            print(f"Loading LoRA adapters from: {model_path}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
                token=hf_token,
            )
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            print(f"Loading full model from: {model_path}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
                token=hf_token,
            )
        
        FastLanguageModel.for_inference(self.model)
        print(f"Mistral classifier loaded for {dataset} dataset!\n")
    
    def _format_prompt(self, text):
        """Format prompt based on dataset type"""
        if self.task_type == 'sentiment':
            prompt = f"Label the following input as one of the following labels: Positive, Negative.\n\nReview: {text}"
        else:  # topic classification
            labels_str = ", ".join([l.capitalize() for l in self.labels])
            prompt = f"Label the following text as one of the following categories: {labels_str}.\n\nText: {text}"
        
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def predict_logits(self, text):
        """Predict using logits"""
        formatted_prompt = self._format_prompt(text)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get logits for the last token position
        score_start = logits[:, -1, :]
        scores = {}
        
        # Extract scores for each label
        for label in self.label_map.values():
            cur_id = self.tokenizer.convert_tokens_to_ids(label)
            
            if cur_id == self.tokenizer.unk_token_id or cur_id is None:
                encoded = self.tokenizer.encode(label, add_special_tokens=False)
                cur_id = encoded[0] if len(encoded) > 0 else self.tokenizer.unk_token_id
            
            scores[label] = score_start[0, cur_id].item()
        
        # Find label with highest score
        pred_label_str = max(scores.keys(), key=lambda x: scores[x])
        pred_label_idx = self.reverse_label_map[pred_label_str]
        confidence = scores[pred_label_str]
        
        return pred_label_idx, confidence, scores


class HQAAttackMistralLogits:
    """HQA-Attack using logits-based prediction for Mistral"""
    
    def __init__(self, model_path,
                 base_model="unsloth/mistral-7b-instruct-v0.3",
                 dataset="rotten_tomatoes",
                 synonym_method='wordnet',
                 embedding_path=None,
                 max_seq_length=512,
                 hf_token=None,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.query_count = 0
        self.dataset = dataset
        
        # Initialize Mistral classifier with logits
        self.classifier = MistralClassifierLogits(
            model_path=model_path,
            base_model=base_model,
            max_seq_length=max_seq_length,
            hf_token=hf_token,
            device=device,
            dataset=dataset
        )
        
        self.label_map = self.classifier.label_map
        
        # Initialize synonym extractor
        self.synonym_extractor = SynonymExtractor(
            method=synonym_method,
            embedding_path=embedding_path
        )
        
        print(f"HQA Attack (Logits) initialized for Mistral on {dataset}")
        print(f"Device: {device}, max_sequence length: {max_seq_length}")
    
    def get_prediction(self, text):
        """Get model prediction using logits"""
        self.query_count += 1
        label_idx, confidence, scores = self.classifier.predict_logits(text)
        return label_idx, confidence
    
    def get_synonyms(self, word, top_k=50):
        return self.synonym_extractor.get_synonyms(word, top_k)
    
    def get_important_words(self, text, pos_tags=['NN', 'VB', 'JJ', 'RB']):
        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            important = []
            for token, tag in tagged:
                if any(tag.startswith(p) for p in pos_tags) and len(token) > 2:
                    important.append((token, tag))
            return important
        except:
            return []
    
    def calculate_similarity(self, text1, text2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1:
            return 0
        intersection = len(words1.intersection(words2))
        return intersection / len(words1)
    
    def substitute_words_back(self, original_text, current_text, original_label, max_iterations=5):
        current_words = current_text.split()
        original_words = original_text.split()
        
        if len(current_words) != len(original_words):
            return current_text
        
        for iteration in range(max_iterations):
            diff_positions = [i for i in range(len(current_words)) 
                             if current_words[i] != original_words[i]]
            
            if not diff_positions:
                break
            
            best_pos = None
            best_sim = -1
            
            for pos in diff_positions:
                test_words = current_words.copy()
                test_words[pos] = original_words[pos]
                test_text = ' '.join(test_words)
                
                pred_label, _ = self.get_prediction(test_text)
                if pred_label != original_label:
                    similarity = self.calculate_similarity(original_text, test_text)
                    if similarity > best_sim:
                        best_sim = similarity
                        best_pos = pos
            
            if best_pos is not None:
                current_words[best_pos] = original_words[best_pos]
            else:
                break
        
        return ' '.join(current_words)
    
    def find_best_replacement(self, word, candidates, adversarial_text, 
                             original_label, position):
        words = adversarial_text.split()
        best_word = word
        best_score = -1
        
        for candidate in candidates:
            test_words = words.copy()
            test_words[position] = candidate
            test_text = ' '.join(test_words)
            
            pred_label, score = self.get_prediction(test_text)
            
            if pred_label != original_label and score > best_score:
                best_score = score
                best_word = candidate
        
        return best_word
    
    def optimize_adversarial(self, original_text, adversarial_text, 
                            original_label, n_samples=5):
        words = adversarial_text.split()
        original_words = original_text.split()
        
        changed_positions = [i for i in range(min(len(words), len(original_words))) 
                            if words[i] != original_words[i]]
        
        if not changed_positions:
            return adversarial_text
        
        for pos in changed_positions:
            current_word = words[pos]
            synonyms = self.get_synonyms(current_word, top_k=50)
            
            if not synonyms:
                continue
            
            candidates = np.random.choice(synonyms, 
                                         min(n_samples, len(synonyms)), 
                                         replace=False).tolist()
            candidates.append(current_word)
            
            best_replacement = self.find_best_replacement(
                current_word, candidates, adversarial_text, original_label, pos
            )
            
            words[pos] = best_replacement
            adversarial_text = ' '.join(words)
        
        return adversarial_text
    
    def initialize_adversarial(self, text, original_label, replacement_pct=0.5, max_attempts=1000):
        words = text.split()
        important = self.get_important_words(text)
        
        if not important:
            important = [(w, 'NOUN') for w in words if len(w) > 2]
        
        if not important or len(important) < 1:
            print(f"WARNING: No important words found in text: {text[:100]}")
            return None
        
        num_replacements = max(1, int(len(important) * replacement_pct))
        print(f"Replacing {num_replacements} out of {len(important)} important words ({replacement_pct*100}%)")
        
        for attempt in range(max_attempts):
            test_words = words.copy()
            selected_important = np.random.choice(len(important), 
                                                num_replacements, 
                                                replace=False)
            
            replaced_count = 0
            for idx in selected_important:
                word, _ = important[idx]
                synonyms = self.get_synonyms(word, top_k=50)
                
                if synonyms:
                    word_positions = [i for i, w in enumerate(test_words) if w.lower() == word.lower()]
                    if word_positions:
                        replacement = np.random.choice(synonyms)
                        pos = np.random.choice(word_positions)
                        test_words[pos] = replacement
                        replaced_count += 1
            
            if replaced_count == 0:
                continue
            
            test_text = ' '.join(test_words)
            pred_label, pred_score = self.get_prediction(test_text)
            
            if pred_label != original_label:
                return test_text
        
        return None
    
    def attack(self, text, true_label=None, max_iterations=5, verbose=True):
        """Execute full HQA-Attack algorithm using logits"""
        self.query_count = 0
        attack_start_time = time.time()
        
        if verbose:
            print(f"\nOriginal: {text[:100]}...")
        
        original_label, original_score = self.get_prediction(text)
        
        # NEW: Check if model's prediction matches ground truth
        if true_label is not None and original_label != true_label:
            if verbose:
                print(f"SKIPPED: Model misclassified originally")
                print(f"True label: {self.label_map.get(true_label, f'Label_{true_label}')}")
                print(f"Predicted: {self.label_map.get(original_label, f'Label_{original_label}')}")
            
            attack_time = time.time() - attack_start_time
            return {
                'original': text,
                'adversarial': None,
                'original_pred_label': original_label,
                'true_label': true_label,
                'final_label': None,
                'success': False,
                'skipped': True,
                'reason': 'original_misclassification',
                'queries': self.query_count,
                'attack_time': attack_time,
                'method': 'logits'
            }
        
        if verbose:
            label_name = self.label_map.get(original_label, f"Label_{original_label}")
            print(f"Label: {label_name} (confidence: {original_score:.3f})")
        
        adversarial = self.initialize_adversarial(text, original_label, 0.8)
        if adversarial is None:
            if verbose:
                print("Failed to initialize adversarial example")
            attack_time = time.time() - attack_start_time
            return {
                'original': text,
                'adversarial': None,
                'original_pred_label': original_label,
                'true_label': true_label,
                'final_label': None,
                'success': False,
                'skipped': False,
                'queries': self.query_count,
                'attack_time': attack_time,
                'method': 'logits'
            }
        
        pred_label, pred_score = self.get_prediction(adversarial)
        
        if verbose:
            label_name = self.label_map.get(pred_label, f"Label_{pred_label}")
            print(f"After init: {adversarial[:100]}...")
            print(f"Pred: {label_name} (confidence: {pred_score:.3f})")
        
        for iteration in range(max_iterations):
            adversarial = self.substitute_words_back(text, adversarial, original_label)
            pred_label, pred_score = self.get_prediction(adversarial)
            
            if pred_label == original_label:
                if verbose:
                    print("Lost adversarial property after word substitution")
                break
            
            adversarial = self.optimize_adversarial(text, adversarial, original_label)
            pred_label, pred_score = self.get_prediction(adversarial)
            
            if pred_label == original_label:
                if verbose:
                    print("Lost adversarial property after optimization")
                break
        
        attack_time = time.time() - attack_start_time
        
        if verbose:
            print(f"Final: {adversarial[:100]}...")
            pred_label, pred_score = self.get_prediction(adversarial)
            label_name = self.label_map.get(pred_label, f"Label_{pred_label}")
            print(f"Final pred: {label_name} (confidence: {pred_score:.3f})")
            print(f"Queries used: {self.query_count}")
            print(f"Attack time: {attack_time:.2f}s")
        
        return {
            'original': text,
            'adversarial': adversarial,
            'original_pred_label': original_label,
            'true_label': true_label,
            'final_label': pred_label,
            'success': pred_label != original_label,
            'skipped': False,
            'queries': self.query_count,
            'attack_time': attack_time,
            'method': 'logits'
        }


def load_checkpoint(checkpoint_file):
    """Load existing checkpoint if available"""
    if os.path.exists(checkpoint_file):
        print(f"\n{'='*80}")
        print(f"RESUMING FROM CHECKPOINT: {checkpoint_file}")
        print(f"{'='*80}\n")
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


def main():
    parser = argparse.ArgumentParser(description="HQA Attack on Mistral using Logits")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="unsloth/mistral-7b-instruct-v0.3")
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
        "yelp_polarity": ("text-classification", "randellcotta/distilbert-base-uncased-finetuned-yelp-polarity", "yelp_polarity", "test"),
        "rotten_tomatoes": ("text-classification", "textattack/distilbert-base-uncased-rotten-tomatoes", "cornell-movie-review-data/rotten_tomatoes", "test")
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
    attacker = HQAAttackMistralLogits(
        model_path=args.model_path,
        base_model=args.base_model,
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
        
        print(f"\n{'='*80}")
        print(f"Sample {current_idx}/{batch_start + total_samples - 1}")
        elapsed_time = time.time() - total_start_time + previous_time
        avg_time = np.mean(sample_times) if sample_times else 0
        remaining = (len(test_data) - i) * avg_time
        print(f"Elapsed: {format_time(elapsed_time)} | "
              f"Avg/sample: {avg_time:.1f}s | "
              f"ETA: {format_time(remaining)}")
        print(f"{'='*80}")
        
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
            print(f"\n{'─'*80}")
            print(f"Sample {i+1} completed in {sample_time:.2f}s")
            if result.get('skipped', False):
                print(f"Status: SKIPPED (original misclassification)")
            else:
                print(f"Success: {result['success']} | Queries: {result['queries']}")
            
            valid_attempts = (i + 1) - skipped_count
            if valid_attempts > 0:
                attack_success_rate = success_count / valid_attempts * 100
                print(f"Attack success rate: {success_count}/{valid_attempts} ({attack_success_rate:.2f}%)")
            print(f"Overall stats: {success_count} success, {failed_count} failed, {skipped_count} skipped")
            print(f"{'─'*80}")
            
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
            print(f"\n[ERROR] Failed on sample {i+1}: {str(e)}")
            # Save checkpoint on error
            current_time = time.time() - total_start_time + previous_time
            save_checkpoint(
                args.checkpoint_file, 
                results, 
                current_idx-1,
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
    
    print(f"\n{'='*80}")
    print(f"ATTACK CAMPAIGN COMPLETED")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Method: Logits")
    print(f"Total samples processed: {len(test_data)}")
    print(f"{'='*80}")
    print(f"ATTACK RESULTS")
    print(f"{'='*80}")
    print(f"Successful attacks: {success_count}")
    print(f"Failed attacks: {failed_count}")
    print(f"Skipped (originally misclassified): {skipped_count}")
    print(f"Valid attack attempts: {valid_attempts}")
    if valid_attempts > 0:
        print(f"Attack success rate: {success_count / valid_attempts * 100:.2f}% ({success_count}/{valid_attempts})")
    print(f"{'='*80}")
    print(f"TIMING STATISTICS")
    print(f"{'='*80}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Average time per sample: {avg_time_per_attack:.2f}s")
    print(f"Fastest attack: {min(sample_times) if sample_times else 0:.2f}s")
    print(f"Slowest attack: {max(sample_times) if sample_times else 0:.2f}s")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_file}")
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)
        print(f"Checkpoint file removed: {args.checkpoint_file}")


if __name__ == "__main__":
    main()