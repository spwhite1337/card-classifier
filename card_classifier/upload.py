import os
import argparse
from config import Config, logger


def upload():
    parser = argparse.ArgumentParser(prog='Upload card classifier')
    parser.add_argument('--skipdata', action='store_true')
    parser.add_argument('--skipresults', action='store_true')
    args = parser.parse_args()

    if not args.skipdata:
        logger.info('Uploading Data')
        os.system('aws s3 sync {} {} --exclude .gitignore'.format(Config.DATA_DIR, Config.CLOUD_DATA))
    if not args.skipresults:
        logger.info('Uploading Results')
        os.system('aws s3 sync {} {} --exclude .gitignore'.format(Config.RESULTS_DIR, Config.CLOUD_RESULTS))
