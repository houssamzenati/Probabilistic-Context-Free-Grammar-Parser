import string
import random
import numpy as np

class OovModule:
    """Class for Out-of-Vocabulary Module

    """
    def __init__(self, vocabulary, embeddings):
        """Initializes the class

        Attributes:
            vocabulary (set): set of words from the vocabulary
            embeddings (np.array): embeddings for the corresponding vocabulary
            word2idx (dict): get index of a word
            idx2word (dict): get word of an index
            charset (str): latin charset

        """
        self.vocabulary = set(vocabulary)
        self.embeddings = embeddings
        self.word2idx = {w: i for (i, w) in enumerate(vocabulary)}
        self.idx2word = dict(enumerate(vocabulary))
        self.charset = string.ascii_lowercase + "àâôéèëêïîçùœ"

    @staticmethod
    def delete(word, pos):
        """ Deletes the char in the word at the position pos
        """
        return word[:pos] + word[pos + 1:]

    @staticmethod
    def insert(word, char, pos):
        """ Inserts the char in the word at the position pos
        """
        return word[:pos] + char + word[pos:]

    @staticmethod
    def substitute(word, char, pos):
        """ Subtitutes the char in the word at the position pos
        """
        return word[:pos] + char + word[pos + 1:]

    def _generate_candidates(self, words, k):
        """Generates all possible candidates for the unrecognized words

        Args:
            words (list): words that are not recognized
            k (int): maximum levenshtein distance authorized in misspelling

        Returns:
            (list) possible candidates
        """
        candidates = []
        for word in words:
            for i, char in enumerate(word):
                candidates += [self.delete(word, i)]
                for char in self.charset:
                    candidates += [self.insert(word, char, i)]
                    candidates += [self.substitute(word, char, i)]
        if k > 1:
            candidates += self._generate_candidates(candidates, k - 1)
        return candidates

    def get_levenshtein_neighbors(self, word, k):
        """Gets all misspelled candidates in the range of known
        vocabulary

        Args:
            word (str): input word
            k (int): maximum levenshtein distance authorized in misspelling

        Returns:
            (set) set of possible neighbors
        """
        candidates = self._generate_candidates([word], k)
        candidates = set(candidates)
        return candidates.intersection(self.vocabulary)

    def _get_cosine_similarities(self, vec):
        """Computes the cosine similarities between the current embedding and all
        other embeddings

        Args:
            vec (np.array): embedding of the current word

        Returns:
            (np.array) all cosine similarities
        """
        inners = np.inner(self.embeddings, vec)
        vec_norm = np.linalg.norm(vec)
        embedding_norms = np.linalg.norm(self.embeddings, axis=1)
        return inners / (vec_norm * embedding_norms)

    def get_embedding_neighbors(self, word, n):
        """Gets most similar words in the range of known
        vocabulary based on cosine similarity with all
        embeddings

        Args:
            word (str): input word
            n (int):  maximum number of closets embeddings authorized

        Note:
            If the word is not in the vocabulary, returns < UNK >
        """
        if word in self.word2idx.keys():
            vec = self.embeddings[self.word2idx[word]]
        else:
            return [self.idx2word[0]]

        similarities = self._get_cosine_similarities(vec)
        candidates_idx = np.argsort(similarities)[-n:]
        candidates_idx = np.flip(candidates_idx)
        return set([self.idx2word[i] for i in candidates_idx])

    def get_neighbor(self, word, k=1, n=5):
        """ Gets most likely neighbors for the input word

        Args:
            word (str): word for which to find neighbors
            k (int): maximum levenshtein distance authorized in misspelling
            n (int): maximum number of closets embeddings authorized

        Note:
            levenshtein and embedding candidates are necessarily in the range
            of know vocabulary (see methods get_levensthein_neighbors and
            definition of the embedding which is intersected with the lexicon)

        Returns
            (str) the most likely word
        """
        levenshtein_candidates = self.get_levenshtein_neighbors(word, k)
        embedding_candidates = self.get_embedding_neighbors(word, n)
        for embedding_candidate in embedding_candidates:
            if embedding_candidate in levenshtein_candidates:
                return embedding_candidate
        if levenshtein_candidates:
            return random.choice(list(levenshtein_candidates))
        else:
            return random.choice(list(embedding_candidates))



