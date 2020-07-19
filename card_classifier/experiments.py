import os
import json
import argparse

from card_classifier.classifier import MagicCardClassifier

from config import Config, logger


def run_experiments():
    parser = argparse.ArgumentParser(prog='Card Classifier Experiments')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--version', type=str, default='v0')
    parser.add_argument('--overwrite', action='store_true')
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
            # Skip if you aren't overwriting and model already exists
            if os.path.exists(os.path.join(Config.RESULTS_DIR, experiment['model_type'], args.version, color)) and not \
                    args.overwrite:
                continue
            logger.info('Fitting {}'.format(color))
            mcc = MagicCardClassifier(version=args.version, train_color=color, **experiment)
            try:
                mcc.train()
                mcc.diagnose()
                mcc.save()
            except Exception as err:
                logger.error(err)
