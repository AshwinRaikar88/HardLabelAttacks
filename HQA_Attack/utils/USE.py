import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from typing import Union, List

class USEncoder:
    def __init__(self, model_url: str = "https://tfhub.dev/google/universal-sentence-encoder-large/5"):
        # Enable GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

        # Load as Keras layer (works with model building)
        self.embed = hub.KerasLayer(model_url)
        print(f"USE model loaded from {model_url}")

    def compute_similarity(self, text1: Union[str, List[str]], text2: Union[str, List[str]]) -> Union[float, np.ndarray]:
        """
        Compute similarity score(s) between text pairs using the exact original formula:
        1. L2 normalizes embeddings
        2. Computes cosine similarity
        3. Clips to [-1, 1] range
        4. Converts to angular distance (1 - acos)
        """
        single_input = isinstance(text1, str)
        if single_input:
            text1 = [text1]
            text2 = [text2]

        # Step 1: Get and normalize embeddings
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        norm1 = tf.nn.l2_normalize(emb1, axis=1)
        norm2 = tf.nn.l2_normalize(emb2, axis=1)

        # Step 2: Cosine similarity
        cosine_sim = tf.reduce_sum(tf.multiply(norm1, norm2), axis=1)

        # Step 3: Clip values
        clipped = tf.clip_by_value(cosine_sim, -1.0, 1.0)

        # Step 4: Angular distance
        sim_scores = 1.0 - tf.acos(clipped)

        return sim_scores.numpy()[0] if single_input else sim_scores.numpy()

# Example usage
if __name__ == "__main__":
    encoder = USEncoder()
    score = encoder.compute_similarity("hello world", "hi universe")
    print(f"Similarity score: {score}")
