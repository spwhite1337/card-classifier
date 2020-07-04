import os
import shutil
import cv2
import random
import argparse

import numpy as np
from tqdm import tqdm

from config import ROOT_DIR, logger

random.seed(187)


def _make_dir(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _image_cropper(fn: str) -> np.ndarray:
    """
    load and crop an image down so that only the artwork shows
    """
    uncropped = cv2.imread(fn)
    h, w = uncropped.shape[:2]
    return uncropped[int(h/10):int(h/1.8), int(w/10):int(9*w/10), :]


def crop_images(colors: list, raw_dir: str, cropped_dir: str):
    """
    Crop raw images to only the artwork
    """
    # Iterate through colors
    for color in tqdm(colors):
        logger.info('Cropping {} images.'.format(color))
        load_dir = os.path.join(raw_dir, color)
        save_dir = _make_dir(os.path.join(cropped_dir, color))

        # Load all images
        for img in tqdm([f for f in os.listdir(load_dir) if '.jpg' in f]):
            # Load and crop image
            cropped = _image_cropper(os.path.join(load_dir, img))
            # Save
            cv2.imwrite(os.path.join(save_dir, img), cropped)


def sort_images(colors: list, cropped_dir: str, curated_dir: str):
    """
    Copy cropped images to labeled dirs
    """
    # Set up training directories with Color vs. NotColor
    for save_color in colors:
        logger.info('Sorting {} Images'.format(save_color))
        pos_dir = _make_dir(os.path.join(curated_dir, save_color, 'train', 'positive'))
        neg_dir = _make_dir(os.path.join(curated_dir, save_color, 'train', 'negative'))

        # Loop through colors, if it is the target color, save those files to the positive dir, else to the negative
        for load_color in tqdm(colors):
            src_dir = os.path.join(cropped_dir, load_color)
            dst_dir = pos_dir if load_color == save_color else neg_dir
            os.system('cp -r {}/*.jpg {}'.format(src_dir, dst_dir))


def split_images(colors: list, curated_dir: str, split: float):
    """
    Move files from training to test
    """
    for split_color in colors:
        logger.info('Moving {} images'.format(split_color))
        for cls in ['positive', 'negative']:
            # Get list of all files
            src_dir = os.path.join(curated_dir, split_color, 'train', cls)
            dst_dir = _make_dir(os.path.join(curated_dir, split_color, 'test', cls))

            # Subset based on split
            src_files = [f for f in os.listdir(src_dir) if '.jpg' in f]
            files_to_move = random.sample(src_files, int(split * len(src_files)))

            # Destination
            for f in tqdm(files_to_move):
                src_file = os.path.join(src_dir, f)
                dst_file = os.path.join(dst_dir, f)
                shutil.move(src_file, dst_file)


def count_cards(curated_dir: str = None):
    """
    Count cards in sorted directories
    """
    if curated_dir is None:
        curated_dir = os.path.join(ROOT_DIR, 'data', 'curated')
    if not os.path.exists(curated_dir):
        return

    for card_dir in os.listdir(curated_dir):
        for target_dir in ['train', 'test']:
            for cls_dir in ['positive', 'negative']:
                logger.info('Num Cards in {}, {}, {}: {}'.format(
                    card_dir, target_dir, cls_dir,
                    len(os.listdir(os.path.join(curated_dir, card_dir, target_dir, cls_dir))))
                )


def curate_images():
    """
    Load images, crop, and save in sorted folder
    """
    # Parse args
    parser = argparse.ArgumentParser(prog='Curation of MTG images')
    parser.add_argument('--split', type=float, default=0.2)
    args = parser.parse_args()

    # Set up dirs
    RAW_DIR = os.path.join(ROOT_DIR, 'data', 'mtg_images')
    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError('Download raw data first.')
    CROPPED_DIR = _make_dir(os.path.join(ROOT_DIR, 'data', 'cropped'))
    CURATED_DIR = _make_dir(os.path.join(ROOT_DIR, 'data', 'curated'))

    # Get card colors
    card_colors = [d for d in os.listdir(RAW_DIR) if '.csv' not in d]

    # Crop images
    crop_images(card_colors, RAW_DIR, CROPPED_DIR)

    # Sort Images
    sort_images(card_colors, CROPPED_DIR, CURATED_DIR)

    # Split for a test set
    split_images(card_colors, CURATED_DIR, args.split)

    # Count cards at the end for quality assurance
    count_cards(CURATED_DIR)
