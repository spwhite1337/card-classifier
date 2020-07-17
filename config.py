import os
import logging

# Setup logs
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get root dir
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


class Config(object):
    ROOT_DIR = ROOT_DIR
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    RAW_DIR = os.path.join(ROOT_DIR, 'data', 'mtg_images')
    CROPPED_DIR = os.path.join(ROOT_DIR, 'data', 'cropped')
    CURATED_DIR = os.path.join(ROOT_DIR, 'data', 'curated')
    RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
    version = 'v1'
    CLOUD_DATA = 's3://scott-p-white/website/data'
    CLOUD_RESULTS = 's3://scott-p-white/website/results'
