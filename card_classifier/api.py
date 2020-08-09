import pprint
import argparse
from card_classifier.classifier import MagicCardClassifier

from config import Config, logger


def api_cli():
    parser = argparse.ArgumentParser(prog='Predict from Magic Card Classifier')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--version', type=str, default=Config.cc_version)
    parser.add_argument('--model_type', type=str, default='VGG')
    parser.add_argument('--display_output', action='store_true')
    args = parser.parse_args()

    api(version=args.version, model_type=args.model_type, input_path=args.input_path,
        display_output=args.display_output)


def api(version: str, model_type: str, input_path: str, display_output: bool = False):
    """
    Predict from trained models
    """
    # Instantiate classifier and load models
    mcc = MagicCardClassifier(version=version, model_type=model_type, load=True)

    # Predict
    output = mcc.predict(input_path)

    if display_output:
        pp = pprint.PrettyPrinter(indent=4, compact=True)
        logger.info('Output: \n{}'.format(pp.pformat(output)))

    return output
