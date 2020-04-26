import os
import requests
import pandas as pd
from tqdm import tqdm

from config import logger, ROOT_DIR


def get_mtg_metadata() -> dict:
    """
    Download metadata on magic card prints including color and url param
    """
    metadata_url = 'https://www.mtgjson.com/files/AllPrintings.json'
    metadata = requests.get(metadata_url)
    metadata = metadata.json()

    return metadata


def wrangle_mtg_metadata(prints_metadata: dict) -> pd.DataFrame:
    """
    Get a df of image urls, colors, and types from metadata json
    """
    df = []
    for cardset, values in prints_metadata.items():
        cards = values.get('cards')
        if cards is not None:
            # Get colors with N for colorless and M for multi-colored
            colors = []
            for card in cards:
                card_colors = card.get('colors')
                if len(card_colors) == 0:
                    colors.append('N')
                elif len(card_colors) > 1:
                    colors.append('M')
                else:
                    colors.append(card_colors[0])

            # Get type of card
            types = [card.get('type') for card in cards]

            # Get multiverseid
            mvids = [card.get('multiverseId') for card in cards]

            # Get urls
            base_url = 'https://gatherer.wizards.com/Handlers/Image.ashx?multiverseid={}&type=card'
            urls = [base_url.format(int(mvid)) if mvid is not None else None for mvid in mvids]

            # Append to df
            df.append(pd.DataFrame({'multiverseId': mvids, 'url': urls, 'color': colors, 'type': types}))
    df = pd.concat(df)

    # Drop duplicates, mostly lands
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


def download_mtg(metadata: pd.DataFrame):
    """
    Download cards from web and save in data/
    """
    save_dir = os.path.join(ROOT_DIR, 'data', 'mtg_images')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    failed_images = []
    for color, df in tqdm(metadata.groupby('color'), total=len(set(metadata['color']))):
        color_dir = os.path.join(save_dir, color)
        if not os.path.exists(color_dir):
            os.mkdir(color_dir)
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            url = row['url']
            fn = row['multiverseId']
            if url is None:
                continue
            try:
                r = requests.get(url, stream=True)
                with open(os.path.join(color_dir, str(int(fn)) + '.jpg'), 'wb') as handler:
                    handler.write(r.content)
            except Exception as err:
                failed_image = (color, fn, err)
                failed_images.append(failed_image)
    df_failed_images = pd.DataFrame(failed_images, columns=['color', 'multiverseId', 'exception'])

    return df_failed_images


def download_magic():
    logger.info('Downloading Metadata')
    metadata = get_mtg_metadata()
    logger.info('Wrangling Metadata')
    metadata = wrangle_mtg_metadata(metadata)
    logger.info('Downloading Image files')
    df_failed = download_mtg(metadata)
    logger.info('Failed to download {} images.'.format(df_failed.shape[0]))
    logger.info('Saving Metadata')
    metadata.to_csv(os.path.join(ROOT_DIR, 'data', 'metadata.csv'), index=False)
    logger.info('Saving Failed Images info.')
    df_failed.to_csv(os.path.join(ROOT_DIR, 'data', 'failed_images.csv'), index=False)