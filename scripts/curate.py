import os
import cv2

import numpy as np
from tqdm import tqdm

from config import ROOT_DIR, logger


def crop_images(uncropped: np.ndarray) -> np.ndarray:
    """
    crop an image down so that only the artwork shows
    """
    h, w = uncropped.shape[:2]
    return uncropped[int(h/10):int(h/1.8), int(w/10):int(9*w/10), :]


def curate_images():
    """
    Load images, crop, and save in sorted folder
    """
    # Set up dirs
    RAW_DIR = os.path.join(ROOT_DIR, 'data', 'mtg_images')
    CURATED_DIR = os.path.join(ROOT_DIR, 'data', 'curated')
    if not os.path.exists(CURATED_DIR):
        os.mkdir(CURATED_DIR)

    # Get card colors
    card_colors = [d for d in os.listdir(RAW_DIR) if '.csv' not in d]

    # Iterate through colors
    for color in tqdm(card_colors):
        logger.info('Cropping {} images.'.format(color))
        # Load all images
        for img in tqdm([f for f in os.listdir(os.path.join(RAW_DIR, color)) if '.jpg' in f]):
            # Crop image
            cropped = crop_images(cv2.imread(os.path.join(RAW_DIR, color, img)))
            # Save to positive class and all negatives
            pos_dir = os.path.join(os.path.join(CURATED_DIR, color))
            neg_dirs = [os.path.join(os.path.join(CURATED_DIR, 'Not' + c)) for c in card_colors if c != color]
            for save_dir in [pos_dir] + neg_dirs:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                cv2.imwrite(os.path.join(save_dir, img), cropped)


def count_cards():
    CURATED_DIR = os.path.join(ROOT_DIR, 'data', 'curated')
    if not os.path.exists(CURATED_DIR):
        return

    for card_dir in os.listdir(CURATED_DIR):
        logger.info('Num Cards in {}: {}'.format(card_dir, len(os.listdir(os.path.join(CURATED_DIR, card_dir)))))
