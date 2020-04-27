import os
import json
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


def _download_magic(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Download cards from web and save in data/
    """
    save_dir = os.path.join(ROOT_DIR, 'data', 'mtg_images')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Download images based on metadata
    failed_images = []
    for color, df in tqdm(metadata.groupby('color'), total=len(set(metadata['color']))):
        color_dir = os.path.join(save_dir, color)
        if not os.path.exists(color_dir):
            os.mkdir(color_dir)
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
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
    """
    Download raw .jpgs of magic the gathering cards
    """
    logger.info('Downloading Metadata')
    metadata = get_mtg_metadata()
    logger.info('Wrangling Metadata')
    metadata = wrangle_mtg_metadata(metadata)
    logger.info('Downloading Image files')
    df_failed = _download_magic(metadata)
    logger.info('Failed to download {} images.'.format(df_failed.shape[0]))
    logger.info('Saving Metadata')
    metadata.to_csv(os.path.join(ROOT_DIR, 'data', 'mtg_images', 'metadata.csv'), index=False)
    logger.info('Saving Failed Images info.')
    df_failed.to_csv(os.path.join(ROOT_DIR, 'data', 'mtg_images', 'failed_images.csv'), index=False)


def get_pokemon_metadata(repo_dir: str = os.path.join(os.path.dirname(ROOT_DIR), 'pokemon-tcg-data')) -> pd.DataFrame:
    """
    Wrangle jsons from cloned pokemon-tcg-data repo for downloading)
    """
    # Get list of meta data files from clone pokemon repo
    metadata_dir = os.path.join(repo_dir, 'json', 'cards')
    files = [f for f in os.listdir(metadata_dir) if '.json' in f]

    df_metadata = []
    for f in files:
        with open(os.path.join(metadata_dir, f), encoding='utf-8') as handler:
            pokemons = json.load(handler)
        names = [pokemon.get('name') for pokemon in pokemons]
        types = [pokemon.get('types') for pokemon in pokemons]
        imageurls = [pokemon.get('imageUrl') for pokemon in pokemons]
        df_fileset = pd.DataFrame({
            'Fileset': [f] * len(names),
            'Name': names,
            'Type': types,
            'URL': imageurls
        })
        df_metadata.append(df_fileset)
    df_metadata = pd.concat(df_metadata)

    # Wrangle type lists
    def wrangle_types(pokemon_type):
        if isinstance(pokemon_type, list):
            pokemon_type = '_'.join(pokemon_type)
        return pokemon_type
    df_metadata['Type'] = df_metadata['Type'].apply(lambda t: wrangle_types(t))

    # Drop duplicates
    df_metadata = df_metadata.drop_duplicates()

    return df_metadata.reset_index(drop=True)


def _download_pokemon(metadata: pd.DataFrame):
    """
    Download pokemon image data from metadata information
    """
    save_dir = os.path.join(ROOT_DIR, 'data', 'pokemon_images')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Download images based on metadata
    failed_images = []
    for pokemon_type, df_type in tqdm(metadata.groupby('Type'), total=len(set(metadata['Type']))):
        type_dir = os.path.join(save_dir, pokemon_type)
        if not os.path.exists(type_dir):
            os.mkdir(type_dir)
        for idx, row in tqdm(df_type.iterrows(), total=df_type.shape[0]):
            url = row['URL']
            fn = row['Name'] + '_{}'.format(idx)
            if url is None:
                continue
            try:
                r = requests.get(url, stream=True)
                with open(os.path.join(type_dir, str(fn) + '.jpg'), 'wb') as handler:
                    handler.write(r.content)
            except Exception as err:
                failed_image = (pokemon_type, fn, err)
                logger.info(failed_image)
                failed_images.append(failed_image)
    df_failed_images = pd.DataFrame(failed_images, columns=['Type', 'Name', 'Exception'])

    return df_failed_images


def download_pokemon():
    """
    Download pokemon data from tcg project after cloning their repo
    """
    logger.info('Loading Metadata.')
    df_metadata = get_pokemon_metadata()
    logger.info('Download Images')
    df_failed_images = _download_pokemon(df_metadata)
    logger.info('Failed to download {} images'.format(df_failed_images.shape[0]))
    logger.info('Saving Metadata.')
    df_metadata.to_csv(os.path.join(ROOT_DIR, 'data', 'pokemon_images', 'metadata.csv'), index=False)
    logger.info('Saving Failed Images info.')
    df_failed_images.to_csv(os.path.join(ROOT_DIR, 'data', 'pokemon_images', 'failed_images.csv'), index=False)
