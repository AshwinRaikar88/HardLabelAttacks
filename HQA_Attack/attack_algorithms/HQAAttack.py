import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

class HQAAttack:
    """Clean implementation of HQA-Attack algorithm"""

    def __init__(self, model_name="textattack/distilbert-base-uncased-ag-news",
                 device="cuda" if torch.cuda.is_available() else "cpu", hf_token=None):
        self.device = device
        self.model_name = model_name
        self.query_count = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
        self.model.to(device)
        self.model.eval()

        self.classifier = pipeline("text-classification",
                                   model=model_name,
                                   device=0 if device == "cuda" else -1)

        self.label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

        print(f"Model loaded: {model_name}")
        print(f"Device: {device}")

    def get_prediction(self, text):
        """Get model prediction for text"""
        self.query_count += 1
        result = self.classifier(text, truncation=True)[0]
        label_idx = int(result['label'].split('_')[1])
        score = result['score']
        return label_idx, score

    def get_synonyms(self, word, pos=None):
        """Get synonyms for a word using WordNet"""
        synonyms = set()
        for synset in wordnet.synsets(word.lower(), pos=pos):
            for lemma in synset.lemmas():
                if lemma.name() != word.lower():
                    synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)[:15]

    def get_similar_words(self, word):
        """Get similar words (synonyms + lemmas)"""
        similar = set()
        # Get all wordnet forms of the word
        for synset in wordnet.synsets(word.lower()):
            for lemma in synset.lemmas():
                similar.add(lemma.name().replace('_', ' '))

        # Return up to 20 similar words
        similar.discard(word.lower())
        return list(similar)[:20]

    def get_important_words(self, text, pos_tags=['NN', 'VB', 'JJ', 'RB']):
        """Extract important words (nouns, verbs, adjectives, adverbs)"""
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
        """Simple word-based similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1:
            return 0
        intersection = len(words1.intersection(words2))
        return intersection / len(words1)

    def substitute_words_back(self, original_text, current_text, original_label, max_iterations=5):
        """
        Phase 1: Substitute original words back
        Greedily restore words to original while maintaining adversarial property
        """
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
        """Find best replacement word from candidates"""
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
        """
        Phase 2: Optimize adversarial example using synonym set
        """
        words = adversarial_text.split()
        original_words = original_text.split()

        changed_positions = [i for i in range(min(len(words), len(original_words)))
                            if words[i] != original_words[i]]

        if not changed_positions:
            return adversarial_text

        for pos in changed_positions:
            current_word = words[pos]

            synonyms = self.get_synonyms(current_word)

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

    def initialize_adversarial(self, text, original_label, max_attempts=100):
        """
        Random initialization: change important words until adversarial
        """
        words = text.split()
        important = self.get_important_words(text)

        if not important:
            # If no important words found, try all words
            important = [(w, 'NOUN') for w in words if len(w) > 2]

        if not important:
            return None

        for attempt in range(max_attempts):
            test_words = words.copy()

            # More aggressive: replace 2-5 words
            # num_replacements = np.random.randint(2, min(6, len(important) + 1))

            num_replacements = len(important)

            if len(important) > 0:
                selected_important = np.random.choice(len(important), min(num_replacements, len(important)), replace=False)
            else:
                continue

            replaced_count = 0
            for idx in selected_important:
                word, _ = important[idx]
                # Try to get synonyms, if none, try similar words
                synonyms = self.get_synonyms(word)
                if not synonyms:
                    synonyms = self.get_similar_words(word)

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

    def attack(self, text, max_iterations=5, verbose=True):
        """
        Execute full HQA-Attack algorithm
        """
        self.query_count = 0

        if verbose:
            print(f"\nOriginal: {text[:100]}...")

        original_label, original_score = self.get_prediction(text)

        if verbose:
            print(f"Label: {self.label_map[original_label]} (confidence: {original_score:.3f})")

        adversarial = self.initialize_adversarial(text, original_label, max_attempts=100)
        if adversarial is None:
            if verbose:
                print("Failed to initialize adversarial example")
            return {
                'original': text,
                'adversarial': "Failed to initialize",
                'original_label': original_label,
                'final_label': None,
                'success': False,
                'queries': self.query_count
            }

        pred_label, pred_score = self.get_prediction(adversarial)

        if verbose:
            print(f"After init: {adversarial[:100]}...")
            print(f"Pred: {self.label_map[pred_label]} (confidence: {pred_score:.3f})")

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

        if verbose:
            print(f"Final: {adversarial[:100]}...")
            pred_label, pred_score = self.get_prediction(adversarial)
            print(f"Final pred: {self.label_map[pred_label]} (confidence: {pred_score:.3f})")
            print(f"Queries used: {self.query_count}")

        return {
            'original': text,
            'adversarial': adversarial,
            'original_label': original_label,
            'final_label': pred_label,
            'success': pred_label != original_label,
            'queries': self.query_count
        }


