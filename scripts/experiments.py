import os
import json

from modeling.classifier import MagicCardClassifier

from config import logger


def run_experiments():
    # Load experiments
    with open(os.path.join(os.getcwd(), 'experiments.json')) as f:
        experiments = json.load(f)

    # Iterate through experiments
    for experiment in experiments:
        mcc = MagicCardClassifier(**experiment)
        try:
            mcc.train()
            mcc.diagnose()
            mcc.save()
        except Exception as err:
            logger.error(err)
