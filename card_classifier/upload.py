import os
import re
import argparse
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

    include_flag = " --exclude '*/.gitignore'"
    if args.windows:
        include_flag = re.sub("'", "", include_flag)

    if not args.skipdata:
        logger.info('Uploading Data')
        cc_sync = sync_base + data_sync + include_flag
        cc_sync += dryrun_arg if args.dryrun else ''
        logger.info(cc_sync)
        os.system(cc_sync)
    if not args.skipresults:
        logger.info('Uploading Results')
        cc_sync = sync_base + results_sync + include_flag
        cc_sync += dryrun_arg if args.dryrun else ''
        logger.info(cc_sync)
        os.system(cc_sync)
