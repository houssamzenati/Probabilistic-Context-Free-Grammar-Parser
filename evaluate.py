import nltk
nltk.download('punkt')
import numpy as np
from tqdm import tqdm
from PYEVALB import parser as evalbparser
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def get_gold(parser, test_corpus, filename='evaluation_data.gold'):
    """ Saves gold file for pyevalb
    """
    with open(filename, 'w') as f:
        for item in test_corpus:
            gold = parser.tree_to_str(parser.preprocess_tree_from_str(item))
            f.write("%s\n" % gold)

def get_predictions(parser, test_corpus, filename='evaluation_data.parser_output'):
    """ Saves test file for pyevalb
    """
    with open(filename, 'w') as f:
        print('Evaluating parser output...')
        for item in tqdm(test_corpus):
            parsed = parser.parse(tree=item)
            f.write("%s\n" % parsed)
        print('Evaluation of the parser complete...')


def score(true_bracket, proposed_bracket):
    """ Performs evaluation on a single parse tree

    Args:
        true_bracket (str): reference parse tree for the current sentence
        proposed_bracket (str): proposed parse tree for the current sentence

    """
    gold_tree = evalbparser.create_from_bracket_string(true_bracket)
    test_tree = evalbparser.create_from_bracket_string(proposed_bracket)

    # Compute recall and precision for POS tags
    y_true = np.array(gold_tree.poss)
    y_pred = np.array(test_tree.poss)

    y_pred = (y_true == y_pred) * 1
    y_true = np.ones(len(y_true))

    precision, recall, f_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1])
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f_score, accuracy


def evaluation(gold_file_name='evaluation_data.gold', prediction_file_name='evaluation_data.parser_output'):
    """ Performs evaluation on the whole proposed parses and compares them with
    the reference gold file

    Computes mean accuracy, precision, recall, F1 score

    Args:
        gold_file_name (str): path for reference file name
        prediction_file_name (str): path for prediction file name
        result_file_name (str): path for result file name
    """
    gold_corpus = []
    with open(gold_file_name) as f:
        for i, l in enumerate(f):
            gold_corpus.append(l)
    test_corpus = []
    with open(prediction_file_name) as f:
        for i, l in enumerate(f):
            test_corpus.append(l)

    precisions = [];
    recalls = [];
    f_scores = [];
    accuracies = []
    for gold, test in zip(gold_corpus, test_corpus):
        precision, recall, f_score, accuracy = score(gold, test)
        precisions.append(float(precision));
        recalls.append(float(recall))
        f_scores.append(float(f_score[0]));
        accuracies.append(accuracy)

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1_score = np.mean(f_scores)
    mean_accuracy = np.mean(accuracies)

    print('POS precision : {:.2f}%'.format(mean_precision * 100))
    print('POS recall : {:.2f}%'.format(mean_recall * 100))
    print('POS F1 score : {:.2f}%'.format(mean_f1_score * 100))
    print('POS accuracy : {:.2f}%'.format(mean_accuracy * 100))

    print('ごくろうさま')
