import os
import json
import argparse

from card_classifier.classifier import MagicCardClassifier

from config import Config, logger


def run_experiments():
    parser = argparse.ArgumentParser(prog='Card Classifier Experiments')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--version', type=str, default=Config.cc_version)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--color', type=str, required=False)
    args = parser.parse_args()

    # Load experiments
    with open(os.path.join(os.getcwd(), 'card_classifier', 'experiments.json')) as f:
        experiments = json.load(f)

    if args.debug:
        experiments = [{'debug': True, 'model_type': 'VGG'}]
        args.version = 'v0'

    # Iterate through experiments
    for experiment in experiments:
        for color in MagicCardClassifier.card_colors:
            if args.color is not None:
                if color != args.color:
                    logger.info('Skipping {}'.format(color))
                    continue
            # Skip if you aren't overwriting and model already exists
            if os.path.exists(os.path.join(Config.RESULTS_DIR, 'card_classifier', experiment['model_type'],
                                           args.version, color)) and not args.overwrite:
                continue
            logger.info('Fitting {}'.format(color))
            mcc = MagicCardClassifier(version=args.version, train_color=color, **experiment)
            mcc.train()
            mcc.diagnose()
            mcc.save()
