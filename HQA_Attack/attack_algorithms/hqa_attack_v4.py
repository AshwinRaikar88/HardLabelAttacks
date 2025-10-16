import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import warnings
import pickle
from scipy.spatial.distance import cosine
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
    
    @staticmethod
    def load_pickled_embeddings(embedding_path):
        """Load pre-computed pickled embeddings"""
        print(f"Loading pickled embeddings from {embedding_path}...")
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded {len(embeddings)} word vectors")
        return embeddings


class SynonymExtractor:
    """Extract synonyms using different methods"""
    
    def __init__(self, method='wordnet', embedding_path=None):
        """
        Initialize synonym extractor
        method: 'wordnet', 'counter-fitted', 'glove'
        embedding_path: path to embedding file (required for counter-fitted and glove)
        """
        self.method = method
        self.embeddings = None
        
        if method in ['counter-fitted', 'glove']:
            if not embedding_path:
                raise ValueError(f"embedding_path required for {method} method")
            
            if method == 'counter-fitted':
                self.embeddings = EmbeddingLoader.load_counter_fitted(embedding_path)
            else:
                self.embeddings = EmbeddingLoader.load_glove(embedding_path)
            
            # Build reverse vocab for faster lookup
            self.word2idx = {word: idx for idx, word in enumerate(self.embeddings.keys())}
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
            
            # Pre-compute cosine similarity matrix
            print("Pre-computing cosine similarity matrix (this may take a moment)...")
            self.sim_matrix = self._compute_similarity_matrix()
            print("Similarity matrix ready")
        
        print(f"Synonym extractor initialized with method: {method}")
    
    def _compute_similarity_matrix(self):
        """Pre-compute cosine similarity between all word pairs"""
        embeddings_array = np.array([self.embeddings[word] for word in sorted(self.embeddings.keys())])
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)
        
        # Compute similarity matrix
        sim_matrix = np.dot(embeddings_array, embeddings_array.T)
        return sim_matrix
    
    def get_synonyms(self, word, top_k=10):
        """Get top-k synonyms for a word"""
        if self.method == 'wordnet':
            return self._get_wordnet_synonyms(word, top_k)
        elif self.method in ['counter-fitted', 'glove']:
            return self._get_embedding_synonyms(word, top_k)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _get_wordnet_synonyms(self, word, top_k=10):
        """Get synonyms using WordNet"""
        synonyms = set()
        for synset in wordnet.synsets(word.lower()):
            for lemma in synset.lemmas():
                similar_word = lemma.name().replace('_', ' ')
                if similar_word.lower() != word.lower():
                    synonyms.add(similar_word)
        
        return list(synonyms)[:top_k]
    
    def _get_embedding_synonyms(self, word, top_k=10):
        """Get synonyms using embedding-based similarity"""
        word_lower = word.lower()
        
        if word_lower not in self.word2idx:
            return []
        
        word_idx = self.word2idx[word_lower]
        similarities = self.sim_matrix[word_idx]
        
        # Get top-k similar words (excluding the word itself)
        top_indices = np.argsort(-similarities)[1:top_k+1]
        
        synonyms = [self.idx2word[idx] for idx in top_indices if similarities[idx] > 0.5]
        
        return synonyms


class HQAAttack:
    """Clean implementation of HQA-Attack algorithm with multiple embedding options"""
    
    def __init__(self, model_name=None,
                 synonym_method=None,
                 embedding_path=None,
                 label_map=None,
                 hf_token=None,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
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
        
        # Use provided label_map or create default
        self.label_map = label_map if label_map else {}
        
        # Initialize synonym extractor
        self.synonym_extractor = SynonymExtractor(
            method=synonym_method,
            embedding_path=embedding_path
        )
        
        print(f"Model loaded: {model_name}")
        print(f"Device: {device}")
    
    def get_prediction(self, text):
        """Get model prediction for text"""
        self.query_count += 1
        result = self.classifier(text, truncation=True)[0]
        label_idx = int(result['label'].split('_')[1])
        score = result['score']
        return label_idx, score
    
    def get_synonyms(self, word, top_k=10):
        """Get synonyms using the configured method"""
        return self.synonym_extractor.get_synonyms(word, top_k)
    
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
            
            synonyms = self.get_synonyms(current_word, top_k=15)
            
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
    
    def initialize_adversarial(self, text, original_label, max_attempts=1000):
        """
        Random initialization: change important words until adversarial
        """
        words = text.split()
        important = self.get_important_words(text)
        
        if not important:
            important = [(w, 'NOUN') for w in words if len(w) > 2]
        
        if not important or len(important) < 1:
            print(f"WARNING: No important words found in text: {text[:100]}")
            return None
        
        for attempt in range(max_attempts):
            test_words = words.copy()
            
            # Fix: Ensure we have at least 1 important word and valid range
            max_replacements = min(6, len(important) + 1)
            min_replacements = min(1, len(important))  # Start from 1, not 2
            
            # Handle edge case where we only have 1 important word
            if max_replacements <= min_replacements:
                num_replacements = 1
            else:
                num_replacements = np.random.randint(min_replacements, max_replacements)
            
            if len(important) > 0:
                selected_important = np.random.choice(len(important), 
                                                    min(num_replacements, len(important)), 
                                                    replace=False)
            else:
                continue
            
            replaced_count = 0
            for idx in selected_important:
                word, _ = important[idx]
                synonyms = self.get_synonyms(word, top_k=15)
                
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
            label_name = self.label_map.get(original_label, f"Label_{original_label}")
            print(f"Label: {label_name} (confidence: {original_score:.3f})")
        
        adversarial = self.initialize_adversarial(text, original_label)
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
        
        if verbose:
            print(f"Final: {adversarial[:100]}...")
            pred_label, pred_score = self.get_prediction(adversarial)
            label_name = self.label_map.get(pred_label, f"Label_{pred_label}")
            print(f"Final pred: {label_name} (confidence: {pred_score:.3f})")
            print(f"Queries used: {self.query_count}")
        
        return {
            'original': text,
            'adversarial': adversarial,
            'original_label': original_label,
            'final_label': pred_label,
            'success': pred_label != original_label,
            'queries': self.query_count
        }

    