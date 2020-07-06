import pickle
import argparse
from modeling.classifier import MagicCardClassifier


def predict():
    parser = argparse.ArgumentParser(prog='Predict from Magic Card Classifier')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--model_type', type=str, default='VGG')
    parser.add_argument('--save_path', type=str, required=False)
    args = parser.parse_args()

    # Instantiate classifier and load models
    mcc = MagicCardClassifier(version=args.version, model_type=args.model_type, load=True)

    # Predict
    outputs = mcc.predict(args.input_path)
    if args.save_path:
        with open(args.save_path, 'wb') as fp:
            pickle.dump(outputs, fp)
