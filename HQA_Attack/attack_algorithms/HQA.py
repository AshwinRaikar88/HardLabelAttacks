import numpy as np
import random
import tensorflow as tf
import nltk
import copy
from nltk.corpus import wordnet as wn

class USEncoder:
    def __init__(self):
        import tensorflow_hub as hub
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def embed(self, sentences):
        return self.model(sentences)

    def compute_similarity(self, s1, s2):
        embeddings = self.embed([s1, s2])
        emb1, emb2 = embeddings[0], embeddings[1]
        emb1 = tf.nn.l2_normalize(emb1, axis=0)
        emb2 = tf.nn.l2_normalize(emb2, axis=0)
        return tf.reduce_sum(tf.multiply(emb1, emb2)).numpy()

class HQA_Attack:
    def __init__(self, model, debug=False):
        self.DEBUG = debug
        self.query_count = 0
        self.max_iter = 5000
        print("Loaded HQA Attack!")

        self.model = model
        self.use_model = USEncoder()

        vector_file_path = './counter-fitted-vectors.txt'
        self.counter_fitted_embeddings = self.load_word_vectors(vector_file_path)

        self.synonym_cache = {}
        self.vocab_words = list(self.counter_fitted_embeddings.keys())
        self.vocab_embeddings = self.use_model.embed(self.vocab_words)
        self.vocab_embeddings = tf.nn.l2_normalize(self.vocab_embeddings, axis=1)

    @staticmethod
    def load_word_vectors(file_path):
        word_vectors = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                if len(line) > 1:
                    word = line[0]
                    vector = np.array([float(val) for val in line[1:]])
                    word_vectors[word] = vector
        return word_vectors

    def reset_params(self):
        self.query_count = 0

    def get_query_count(self):
        return self.query_count

    def cosine_similarity(self, emb1, emb2):
        emb1 = tf.nn.l2_normalize(emb1, axis=1)
        emb2 = tf.nn.l2_normalize(emb2, axis=1)
        return tf.reduce_sum(tf.multiply(emb1, emb2), axis=1).numpy()

    def get_synonyms_from_embeddings(self, word, top_n=50):
        word_embedding = self.use_model.embed([word])
        word_embedding = tf.nn.l2_normalize(word_embedding, axis=1)
        sim_scores = tf.reduce_sum(tf.multiply(self.vocab_embeddings, word_embedding), axis=1)
        sim_scores = tf.clip_by_value(sim_scores, -1.0, 1.0)
        sim_scores = 1.0 - tf.acos(sim_scores)
        scored_synonyms = list(zip(self.vocab_words, sim_scores.numpy()))
        scored_synonyms.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scored_synonyms[:top_n]]

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                lemm = lemma.name().replace("_", " ")
                if len(lemm.split()) > 1:
                    continue
                synonyms.add(lemm)
        return list(synonyms)

    def replace_synonyms(self, text, label, victim_model, max_iter=100):
        adversarial_example = text.split()
        num_queries = 0
        replaced = set()

        while num_queries < max_iter:
            idx = random.randint(0, len(adversarial_example) - 1)
            word = adversarial_example[idx]
            if idx in replaced:
                continue

            if word not in self.synonym_cache:
                self.synonym_cache[word] = self.get_synonyms_from_embeddings(word)
            synonyms = self.synonym_cache[word]

            for synonym in synonyms:
                if synonym == word or len(synonym.split()) > 1:
                    continue
                temp_text = adversarial_example[:]
                temp_text[idx] = synonym
                temp_str = " ".join(temp_text)

                if victim_model.predict([temp_str])[0] != label:
                    return temp_str, num_queries + 1
                num_queries += 1
                if num_queries >= max_iter:
                    break

            replaced.add(idx)

        return " ".join(adversarial_example), num_queries

    def generate_random_adversarial_example(self, x, orig_label):
      """
      Generate a random adversarial example by replacing certain words with synonyms,
      based on POS tags and model feedback.

      x: input string (original sentence)
      orig_label: label predicted by victim model

      return: new generated adversarial example (string)
      """
      words = x.split()
      adversarial_example = words[:]
      pos_tags = nltk.pos_tag(words)

      target_pos = {'NN', 'VB', 'RB', 'JJ'}
      modified_indices = set()

      while self.query_count < self.max_iter:
          # Find candidate indices for replacement
          candidates = [
              (i, word) for i, (word, pos) in enumerate(pos_tags)
              if pos[:2] in target_pos and i not in modified_indices
          ]

          if not candidates:
              if self.DEBUG:
                  print("No synonyms found or no more candidates.")
              break

          # Randomly select a word to replace
          idx, word_to_replace = random.choice(candidates)

          if word_to_replace not in self.synonym_cache:
              self.synonym_cache[word_to_replace] = self.get_synonyms_from_embeddings(word_to_replace)

          synonyms = self.synonym_cache[word_to_replace]

          # Try replacing with synonyms one by one
          for synonym in synonyms:
              if synonym == word_to_replace or len(synonym.split()) > 1:
                  continue

              temp = adversarial_example[:]
              temp[idx] = synonym
              temp_text = " ".join(temp)

              prediction = self.model(temp_text)[0]['label']
              self.query_count += 1

              if prediction != orig_label:
                  if self.DEBUG:
                      print(f"Replaced '{word_to_replace}' with '{synonym}' at index {idx}")
                  return temp_text

              if self.query_count >= self.max_iter:
                  break

          modified_indices.add(idx)

      if self.DEBUG:
          print(f"No. of queries made = {self.query_count}")
          if self.query_count < self.max_iter:
              print("Adversarial example found!")
          else:
              print("Max iterations reached! No adversarial example found.")

      return " ".join(adversarial_example)

    def substitute_original_words(self, x, x_t, orig_label):
        x = x.split()
        x_t = x_t.split()

        while True:
            diffs = [i for i, (orig, adv) in enumerate(zip(x, x_t)) if orig != adv]
            if not diffs:
                break

            best_choice = None
            best_sim_score = -1
            best_x_tmp = None

            for i in diffs:
                x_tmp = copy.deepcopy(x_t)
                x_tmp[i] = x[i]
                sim_score = self.use_model.compute_similarity(' '.join(x), ' '.join(x_tmp))
                if sim_score > best_sim_score:
                    best_sim_score = sim_score
                    best_choice = i
                    best_x_tmp = x_tmp

            if best_choice is not None:
                if self.model(' '.join(best_x_tmp))[0]['label'] != orig_label:
                    self.query_count += 1
                    x_t = best_x_tmp
                else:
                    break
            else:
                break

            if self.query_count == self.max_iter:
                break

        return ' '.join(x_t)