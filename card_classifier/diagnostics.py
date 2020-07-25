import argparse
from card_classifier.classifier import MagicCardClassifier


def diagnose():
    parser = argparse.ArgumentParser(prog='Diagnostics Report')
    parser.add_argument('--model_type', type=str, default='VGG')
    parser.add_argument('--version', type=str, default='v1')
    args = parser.parse_args()

    # Run diagnostics by loading all models and saving results to an "all" directory
    mcc = MagicCardClassifier(
        model_type=args.model_type,
        version=args.version,
        load=True,
        train_color='all'
    )
    mcc.diagnose()
