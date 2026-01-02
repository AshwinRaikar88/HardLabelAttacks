import torch
import time
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from models.mistral_classsifier import MistralClassifier
from utils.synonym_extractor import SynonymExtractor
from utils.attack_utils import *


class HQAAttack:
    """HQA-Attack using logits-based prediction for Mistral"""

    def __init__(self, model_path,
                 llm_model="mistral",
                 dataset="rotten_tomatoes",
                 synonym_method='counter-fitted',
                 embedding_path=None,
                 hf_token=None,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.query_count = 0
        self.dataset = dataset

        if llm_model == "mistral":
            self.classifier = MistralClassifier(
                model_path=model_path,
                hf_token=hf_token,
                device=device,
                dataset=dataset
            )
        else:
            raise ValueError(f"Unknown LLM model: {llm_model}")

        self.label_map = self.classifier.label_map

        # Initialize synonym extractor
        self.synonym_extractor = SynonymExtractor(
            method=synonym_method,
            embedding_path=embedding_path
        )

        print(f"HQA Attack (Logits) initialized for Mistral on {dataset}")

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
        print(f"Replacing {num_replacements} out of {len(important)} important words ({replacement_pct * 100}%)")

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
