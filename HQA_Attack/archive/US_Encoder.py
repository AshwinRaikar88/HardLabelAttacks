import tensorflow_hub as hub
# import tensorflow as tf
import numpy as np


class US_Encoder:
    """
    A class to hold Universal Sequence Encoder
    
    """
    def __init__(self):            
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        print("Universal Sequence Encoder model loaded!")
    
    def compute_similarity(self, text1, text2):
        """
        Compute the similarity score between two texts using Universal Sentence Encoder (USE).

        text1: First sentence (string)
        text2: Second sentence (string)

        return: Cosine similarity score
        """
        # Encode the sentences into embeddings
        embeddings = self.model([text1, text2])

        # Compute cosine similarity
        similarity = np.inner(embeddings[0], embeddings[1])

        return similarity