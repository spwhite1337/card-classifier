import os
import re
import argparse
import shutil

from tqdm import tqdm

from config import Config, logger


def upload():
    parser = argparse.ArgumentParser(prog='Upload card classifier')
    parser.add_argument('--windows', action='store_true')
    parser.add_argument('--skipdata', action='store_true')
    parser.add_argument('--skipresults', action='store_true')
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()

    # General commands
    sync_base = 'aws s3 sync '
    dryrun_arg = ' --dryrun'
    results_sync = '{} {}'.format(Config.RESULTS_DIR, Config.CLOUD_RESULTS)
    data_sync = '{} {}'.format(Config.DATA_DIR, Config.CLOUD_DATA)
    data_include = " --exclude '*' --include '*.zip' --include 'card_classifier/cc_samples/*'"
    results_include = " --exclude '*/.gitignore' "
    if args.windows:
        data_include = re.sub("'", "", data_include)
        results_include = re.sub("'", "", results_include)

    if not args.skipdata:
        logger.info('Zipping Image data')
        for directory in tqdm(['cropped', 'curated', 'mtg_images']):
            if os.path.exists(os.path.join(Config.DATA_DIR, 'card_classifier', directory)):
                continue
            zip_dir = os.path.join(Config.DATA_DIR, 'card_classifier', directory)
            shutil.make_archive(zip_dir, 'zip', os.path.dirname(zip_dir))

        logger.info('Uploading Data')
        cc_sync = sync_base + data_sync + data_include
        cc_sync += dryrun_arg if args.dryrun else ''
        logger.info(cc_sync)
        os.system(cc_sync)
    if not args.skipresults:
        logger.info('Uploading Results')
        cc_sync = sync_base + results_sync + results_include
        cc_sync += dryrun_arg if args.dryrun else ''
        logger.info(cc_sync)
        os.system(cc_sync)
