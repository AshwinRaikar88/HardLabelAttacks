import copy
import random

import transformers
import random
import nltk

from nltk.corpus import wordnet
# from US_Encoder import US_Encoder
from utils.USE import USE

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class HQA_Algos:
    def __init__(self, model, DEBUG=False):
        self.DEBUG = DEBUG
        self.query_count = 0
        self.max_iter = 100
        print("Loaded HQA Attack!")

        # Sentiment analysis model
        self.model = model

        # Universal Sequence Encoder
        # self.use_model = US_Encoder()
        self.use_model = USE()

        vector_file_path = '/home/raikara/NLP Research/counter-fitted-vectors.txt'
        self.counter_fitted_embeddings = self.load_word_vectors(vector_file_path)

        if self.DEBUG:
            print("Debugging")
    
    @staticmethod
    def load_word_vectors(file_path):
        """Loads word vectors from a text file."""
        word_vectors = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                if len(line) > 1:
                    word = line[0]
                    vector = np.array([float(val) for val in line[1:]])
                    word_vectors[word] = vector
        return word_vectors

    
    @staticmethod
    def get_synonyms(word):
        """
        Retrieve a set of synonyms for a given word using WordNet.

        word: string representing the target word

        return: set of synonyms for the word
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                lemm = lemma.name().replace("_", " ")
                if len(lemm.split()) > 1:
                    # Multi word synonyms are discarded
                    continue
            synonyms.add(lemm)
        return list(synonyms)
    
    def get_synonyms_from_embeddings(self, word, embeddings, top_n=5):
        """
            Gets the top N synonyms for a given word using pre-trained word embeddings.

            Args:
                word (str): The word for which to find synonyms.
                embeddings (dict): A dictionary where keys are words and values are their vector embeddings (NumPy arrays).
                top_n (int): The maximum number of synonyms to return.

            Returns:
                list: A list of tuples, where each tuple contains a synonym and its cosine similarity score,
                    sorted in descending order of similarity. Returns an empty list if the word is not found.
        """
        if word not in embeddings:
            # print(f"Warning: Word '{word}' not found in the embeddings vocabulary.")
            return []

        word_vector = embeddings[word].reshape(1, -1)  # Reshape for cosine_similarity

        synonyms_with_scores = []
        for vocab_word, vocab_vector in embeddings.items():
            if vocab_word != word:  # Exclude the input word itself
                vocab_vector = vocab_vector.reshape(1, -1)
                # similarity_score = cosine_similarity(word_vector, vocab_vector)[0][0]
                sim_score = self.use_model.compute_similarity(word_vector, vocab_vector)                
                synonyms_with_scores.append((vocab_word, sim_score))

        # Sort by similarity score in descending order
        synonyms_with_scores.sort(key=lambda item: item[1], reverse=True)
        # Extract only the synonyms (the first element of each tuple)
        synonyms_list = [synonym for synonym, score in synonyms_with_scores[:top_n]]

        # return synonyms_with_scores[:top_n]
        print(synonyms_list)
        return synonyms_list
    
    def reset_params(self):
        self.query_count = 0

    def replace_synonyms(self, adversarial_example, pos_tags, target_pos):
          """
          Replace words in an adversarial example with their synonyms based on their
          POS tags.

          adversarial_example: randomly initialized adverstial example
          pos_tags: POS tags
          target_pos: set of broad POS categories to be replaced

          return: modified adversarial example
          """
          for i, (word, pos) in enumerate(pos_tags):
            if pos[:2] in target_pos:  # Match broad POS categories
                #   synonyms = self.get_synonyms(word)
                synonyms = self.get_synonyms_from_embeddings(word, self.counter_fitted_embeddings, top_n=5)
                if synonyms:
                    adversarial_example[i] = random.choice(synonyms)  # Random synonym replacement
          return ' '.join(adversarial_example)

    def generate_random_adversarial_example(self, x, orig_label):
        """
        Generate a random adversarial example by replacing certain words with synonyms.

        x: list of words (original text)
        f: victim model (a function that returns model predictions)

        return: new generated adversarial example
        """
        x = x.split()
        # Get Part-Of-Speech (POS) tags for words in x
        pos_tags = nltk.pos_tag(x)
        adversarial_example = x[:]  # Copy the original text        

        # Define POS tags to be replaced (NN: noun, VB: verb, RB: adverb, JJ: adjective)
        target_pos = {'NN', 'VB', 'RB', 'JJ'}

        adv_exmp = self.replace_synonyms(adversarial_example, pos_tags, target_pos)
        
        # Ensure the adversarial condition is met (prediction change)        
        self.query_count += 1

        while orig_label == self.model(adv_exmp)[0]['label']:
            self.query_count += 1
            if self.query_count == self.max_iter:
                break            

            pos_tags = nltk.pos_tag(adversarial_example)
            adv_exmp = self.replace_synonyms(adversarial_example, pos_tags, target_pos)

            if not any(self.get_synonyms(word) for word, pos in pos_tags if pos[:2] in target_pos):
                print("No synonyms found!")
                break

        if self.DEBUG:
          print(f"No. of queries made = {self.query_count}")
          if self.query_count < self.max_iter:
              print("Adversarial example found!")
          else:
              print("Max iterations reached!")
              print("No adversarial example found!")

        return adv_exmp

        
    def substitute_original_words(self, x, x_t, orig_label):
        """
        Improved Algorithm 1: Substituting Original Words Back. New implementation

        x: Original text (list of words)
        x_t: Adversarial example (list of words)
        f: Victim model (a function that returns model predictions)
        compute_similarity: Function to compute similarity between sentences
        query_count: Counter for model queries

        return: New adversarial example x_t after substitution
        """
        
        x = x.split()
        x_t = x_t.split()
        while True:
            diffs = [i for i, (orig, adv) in enumerate(zip(x, x_t)) if orig != adv]
            if not diffs:
                print("No differences remaining.")
                break

            best_choice = None
            best_sim_score = -1
            best_x_tmp = None

            for i in diffs:
                x_tmp = copy.deepcopy(x_t)
                x_tmp[i] = x[i]  # Replace adversarial word with original

                sim_score = self.use_model.compute_similarity(' '.join(x), ' '.join(x_tmp))

                if sim_score > best_sim_score:
                    best_sim_score = sim_score
                    best_choice = i
                    best_x_tmp = x_tmp

            if best_choice is not None:
                # Check if adversarial condition is still met
                if self.model(' '.join(best_x_tmp))[0]['label'] != orig_label:
                    self.query_count += 1
                    x_t = best_x_tmp  # Apply best rollback found
                else:
                    break  # Stop if no more valid replacements can be made
            else:
                break

            if self.query_count == self.max_iter:
                break

        return ' '.join(x_t)
    
