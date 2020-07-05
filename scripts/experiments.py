import os
import json

from modeling.classifier import MagicCardClassifier

from config import logger


def configure_experiments(config: dict) -> list:
    """
    Configure the input json input a list of dictionaries
    """
    return []


def run_experiments():
    # Load experiments
    with open(os.path.join(os.getcwd(), 'experiments.json')) as f:
        experiment_config = json.load(f)

    # Create a list of dictionary for every combination of inputs
    experiments = configure_experiments(experiment_config)

    for experiment in experiments:
        mcc = MagicCardClassifier(**experiment)
        try:
            mcc.train()
            mcc.diagnose()
        except Exception as err:
            logger.error(err)
