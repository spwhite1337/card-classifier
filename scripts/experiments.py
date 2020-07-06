import os
import json
import argparse

from modeling.classifier import MagicCardClassifier

from config import logger


def run_experiments():
    parser = argparse.ArgumentParser(prog='Card Classifier Experiments')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--version', type=str, default='v0')
    args = parser.parse_args()

    # Load experiments
    with open(os.path.join(os.getcwd(), 'scripts', 'experiments.json')) as f:
        experiments = json.load(f)

    if args.debug:
        experiments = [{'debug': True}]
        args.version = 'v0'

    # Iterate through experiments
    for experiment in experiments:
        mcc = MagicCardClassifier(version=args.version, **experiment)
        try:
            mcc.train()
            mcc.diagnose()
            mcc.save()
        except Exception as err:
            logger.error(err)
