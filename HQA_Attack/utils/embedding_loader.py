import numpy as np

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