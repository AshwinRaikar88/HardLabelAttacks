import numpy as np
from embedding_loader import EmbeddingLoader
from nltk.corpus import wordnet

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
        top_indices = np.argsort(-similarities)[1:top_k + 1]
        synonyms = [self.idx2word[idx] for idx in top_indices if similarities[idx] > 0.5]
        return synonyms