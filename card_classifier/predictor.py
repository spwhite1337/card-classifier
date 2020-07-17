import pickle
import pprint
import argparse
from card_classifier.classifier import MagicCardClassifier


def predict_cli():
    parser = argparse.ArgumentParser(prog='Predict from Magic Card Classifier')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--model_type', type=str, default='VGG')
    parser.add_argument('--save_path', type=str, required=False)
    parser.add_argument('--display_output', action='store_true')
    args = parser.parse_args()

    predict(version=args.version, model_type=args.model_type, input_path=args.input_path, save_path=args.save_path,
            display_output=args.display_output)


def predict(version: str, model_type: str, input_path: str, save_path: str = None, display_output: bool = False):
    """
    Predict from trained models
    """
    # Instantiate classifier and load models
    mcc = MagicCardClassifier(version=version, model_type=model_type, load=True)

    # Predict
    outputs = mcc.predict(input_path)

    if display_output:
        pp = pprint.PrettyPrinter(indent=4, compact=True)
        pp.pprint(outputs)

    # Save outputs
    if save_path:
        with open(save_path, 'wb') as fp:
            pickle.dump(outputs, fp)
