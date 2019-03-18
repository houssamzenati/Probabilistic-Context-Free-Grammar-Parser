import os
import argparse
import logging
from pcfg import PCFG
from oov import OovModule
from data import data
from evaluate import get_gold, get_predictions, evaluation

# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logger = logging.getLogger("PCFG Parser")


def run(args):

    has_effect = False

    if args:
        try:

            train_corpus, val_corpus, test_corpus = data.get_train_val_test()
            words, embeddings = data.get_polyglot_words_embeddings()

            parser = PCFG()
            parser.learn_probabilities_and_rules(train_corpus)
            parser.set_oov_module(OovModule, words, embeddings)

            if args.inference:

                get_gold(parser, test_corpus, filename='evaluation_data.gold')
                get_predictions(parser, test_corpus, filename='evaluation_data.parser_output')

            if args.evaluation:
                evaluation('evaluation_data.gold', 'evaluation_data.parser_output')

            if args.parse:
                parser.parse_from_txt(args.txt_path)

        except Exception as e:
            logger.exception(e)
            logger.error("Uhoh, the script halted with an error.")
    else:
        if not has_effect:
            logger.error(
                "Script halted without any effect. To run code, use command:\npython3 main.py <args>")

def path(d):
    try:
        assert os.path.exists(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Run scripts for the NLP assignment')

    argparser.add_argument('--inference', action='store_true',  help='performs inference on the test set')
    argparser.add_argument('--evaluation', action='store_true', help='performs POS evaluation')
    argparser.add_argument('--parse', action='store_true', help='parses .txt file')
    argparser.add_argument('txt_path', nargs="?", type=path, help='path for txt file to be parsed')

    run(argparser.parse_args())


