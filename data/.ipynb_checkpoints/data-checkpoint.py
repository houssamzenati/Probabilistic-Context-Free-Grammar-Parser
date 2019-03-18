import pickle
import numpy
import os

def get_polyglot_words_embeddings(path='data'):
    """ Gets words and embeddings from polyglot lexicon
    """
    full_path = os.path.join(path, 'polyglot-fr.pkl')
    with open(full_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        words, embeddings = u.load()

    return words, embeddings


def get_corpus(path='data'):
    """ Gets whole corpus
    """
    corpus = []
    full_path = os.path.join(path, 'sequoia-corpus+fct.mrg_strict')
    with open(full_path) as f:
        for i, l in enumerate(f):
            corpus.append(l)
    return corpus

def get_train_val_test(path='data'):
    """ Gets training, validation and testing corpus
    """
    corpus = get_corpus(path)
    train_mile = int(len(corpus)*0.8)
    val_mile = int(len(corpus)*0.9)
    train_corpus = corpus[:train_mile]
    val_corpus = corpus[train_mile:val_mile]
    test_corpus = corpus[val_mile:]

    return train_corpus, val_corpus, test_corpus