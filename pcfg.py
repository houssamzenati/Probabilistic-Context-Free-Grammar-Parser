from nltk import Tree, word_tokenize
from nltk.grammar import Production
import numpy as np


class PCFG:
    """Class for Probabilistic Context Free Grammar Parser

    Uses OOV module when set to.

    """
    def __init__(self):
        """Initializes the class

        Attributes:
            rules (dict): dictionary of rules \alpha -> \beta
            non_terminals (dict): dictionary of non terminal symbols
            unary_rules_proba (dict): store unary rules proba \alpha -> \beta
            binary_rules_proba (dict): store binary rules proba \alpha -> \beta_1 \beta_2
            oov_module (OovModule): handle out of vocabulary words

        Note:
            Assumes the tree are in Chomsky Normal Form: see method preprocess_tree_from_str()

        """
        self.rules = {}
        self.non_terminals = {}
        self.unary_rules_proba = {}
        self.binary_rules_proba = {}
        self.transition_rules_list = {}
        self.oov_module = None

    @staticmethod
    def _is_endpoint_tree(subtree):
        """ Checks whether the considered subtree is a leaf
        """
        return type(subtree) == str

    @staticmethod
    def tree_to_str(tree):
        """ Transforms a nltk tree to a string
        """
        return ' '.join(str(tree).split())

    def _update_rules(self, prod):
        """ Update dictionary of rules by counting encounters

        Args:
            prod (nltk.Tree.Production): count the current production

        """
        if prod in self.rules.keys():
            self.rules[prod] += 1
        else:
            self.rules[prod] = 1

    def _update_non_terminals(self, prod):
        """ Update dictionary of non terminals by counting encounters

        Args:
            prod (nltk.Tree.Production): count the current production

        """
        if prod.lhs() in self.non_terminals.keys():
            self.non_terminals[prod.lhs()] += 1
        else:
            self.non_terminals[prod.lhs()] = 1

    def _remove_functional_labels(self, tree):
        """ Ignoring functional labels and removing hyphen in a non-terminal name
        """
        for index, subtree in enumerate(tree):
            if not self._is_endpoint_tree(subtree):
                label = subtree.label().split('-')[0]
                subtree.set_label(label)
                tree[index] = subtree
                self._remove_functional_labels(subtree)
        return tree

    def preprocess_tree_from_str(self, line):
        """ Preprocesses the tree to have Chomsky Normal Form

        Args:
            line (str): current line read from corpus of trees

        Returns:
            tree (nltk.Tree): preprocessed tree
        """
        tree = Tree.fromstring(line, remove_empty_top_bracketing=True)
        tree = self._remove_functional_labels(tree)
        tree.collapse_unary(collapsePOS=True)
        tree.chomsky_normal_form()

        return tree

    def _learn_rules_from_corpus(self, corpus):
        """ Learns dictionary of rules and non terminals from training corpus

        Args:
            corpus (list): list of training trees

        """
        for line in corpus:
            tree = self.preprocess_tree_from_str(line)
            for prod in tree.productions():
                self._update_rules(prod)
                self._update_non_terminals(prod)

    def learn_probabilities_and_rules(self, corpus):
        """ Learns dictionary of unary and binary probabilities from training corpus

        Args:
            corpus (list): list of training trees

        """
        print('Learning probability rules from training corpus...')
        self._learn_rules_from_corpus(corpus)
        for symbol in self.rules.keys():

            lhs = symbol.lhs()
            rhs = symbol.rhs()
            is_unary = len(rhs) == 1
            if is_unary:
                self.unary_rules_proba[symbol] = float(self.rules[symbol] / self.non_terminals[lhs])
            else:
                self.binary_rules_proba[symbol] = float(self.rules[symbol] / self.non_terminals[lhs])

            if lhs not in self.transition_rules_list.keys():
                self.transition_rules_list[lhs] = []
            if len(rhs) == 2:
                self.transition_rules_list[lhs].append(rhs)

        print('Probabilities learned from training corpus!')

    def retrieve_tree(self, tokens, bp, i, j, X):
        """ Get the parsed tree with back pointers.

        Args:
            tokens (list): list of str tokens
            bp (dict): dictionary of back pointers
            i (int): i-th element of the token list
            j (int): j-th element of the token list

        Returns:
            (str): parsed tree in str form
        """
        if i == j:
            return "".join([str(X), ' ', str(tokens[i])])
        else:
            Y, Z, s = bp[i, j, X]
            return "".join([str(X),
                            ' (',
                            self.retrieve_tree(tokens, bp, i, s, Y),
                            ') (',
                            self.retrieve_tree(tokens, bp, s + 1, j, Z), ')'])

    def retrieve_lexicon(self):
        """ Retrieves the lexicon learned from the training corpus
        """
        return [symbol.rhs()[0] for symbol in self.rules.keys() if self._is_endpoint_tree(symbol.rhs()[0])]


    def CYK(self, tokens):
        """ Performs the Cocke–Younger–Kasami using dynamic programming

        Args:
            tokens (list): list of str tokens

        Note:
            Pseudo code of this algorithm available at:
            http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf

        Returns:
            (str): most probable parsing tree spanning the tokens
        """
        n = len(tokens)
        pi = {}
        bp = {}
        N = self.non_terminals.keys()
        retrieved_lexicon = self.retrieve_lexicon()

        # Initialization of the probability table
        for i in range(n):
            token = tokens[i]
            if token in retrieved_lexicon:
                w = token
            else:
                if self.oov_module:
                    w = self.oov_module.get_neighbor(token)
                    tokens[i] = w
                else:
                    w = '<UNK>'

            for X in N:
                if Production(X, (w,)) in self.rules:
                    pi[i, i, X] = self.unary_rules_proba[Production(X, tuple((w,)))]
                else:
                    pi[i, i, X] = 0

        # Algorithm
        for l in range(1, n):
            for i in range(n - l):
                j = i + l
                for X in N:
                    max_score = 0
                    args = None
                    for R in self.transition_rules_list[X]:
                        Y, Z = R[0], R[1]
                        for s in range(i, j):
                            if pi[i, s, Y] > 0 and pi[s + 1, j, Z] > 0:
                                score = self.binary_rules_proba[Production(X, R)] * pi[i, s, Y] * pi[s + 1, j, Z]
                                if max_score < score:
                                    max_score = score
                                    args = Y, Z, s

                    if max_score > 0:
                        bp[i, j, X] = args

                    pi[i, j, X] = max_score

        # Retrieve the most probable parsed tree from backpointers and argmax of probability table
        max_score = 0
        args = None
        for X in N:

            if max_score < pi[0, n - 1, X]:
                max_score = pi[0, n - 1, X]
                args = 0, n - 1, X

        if args == None:
            return '(SENT (UNK))'
        else:
            return '(SENT (' + self.retrieve_tree(tokens, bp, *args) + '))'

    def set_oov_module(self, oovmodule, vocabulary, embeddings):
        """ Sets the OOV module that assigns a (unique) part-of-speech to any token
        not included in the lexicon extracted from the training corpus.

        Args:
            oovmodule (OovModule): the oov module to be initialized
            vocabulary (list): list of words from the Polyglot embedding for French
            embeddings (np.array): embeddings of the vocabulary list from the Polyglot embedding for French
                                   of shape ([None, 64])

        Note:
            Finds intersection between learned lexicon and the Polyglot lexicon in order to get embeddings
            of the learned lexicon. Then initializes the OOV module

        """
        lexicon = self.retrieve_lexicon()
        intersection = set(lexicon).intersection(set(vocabulary))
        word2idx = {w: i for (i, w) in enumerate(vocabulary)}
        mask_indexes = [word2idx[word] for word in intersection]
        new_vocabulary = np.array(vocabulary)[mask_indexes].tolist()
        new_embeddings = embeddings[mask_indexes]

        self.oov_module = oovmodule(new_vocabulary, new_embeddings)
        print('Oov Module is set to work!')

    def tokenize_word_for_tree(self, line):
        """ Tokenizes the line read from tree

        Args:
            line (str): current line read from corpus of trees

        Returns:
            (list): list of tokens
        """
        return self.preprocess_tree_from_str(line).leaves()

    def parse(self, sentence=None, tree=None):
        """ Parses sentences or tree like str

        Args:
            sentence (str): if the str to be parsed is natural language
            tree (str): if the str to be parsed is in bracket format

        """
        if sentence:
            tokens = word_tokenize(sentence)
        elif tree:
            tokens = self.tokenize_word_for_tree(tree)
        else:
            print('Not possible to parse this!')
            raise ValueError
        return self.CYK(tokens)

    def parse_from_txt(self, txt_path):
        """ Parses from txt file
        """
        file = []
        with open(txt_path) as f:
            for l in f:
                file.append(l)

        for line in file:
            print(self.parse(sentence=line))
