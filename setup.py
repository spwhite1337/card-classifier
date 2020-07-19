from setuptools import setup, find_packages

setup(
    name='card-classifier',
    version='1.0',
    description='Quiz Backend for playing card classification',
    author='Scott P. White',
    author_email='spwhite1337@gmail.com',
    packages=find_packages(),
    entry_points={'console_scripts': [
        'cc_download = card_classifier.download:download_magic',
        'cc_curate_images = card_classifier.curate:curate_images',
        'cc_count_cards = card_classifier.curate:count_cards',
        'cc_run_experiments = card_classifier.experiments:run_experiments',
        'cc_predictions = card_classifier.api:api_cli',
        'cc_upload = card_classifier.upload:upload'
    ]},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'requests',
        'plotly',
        'dash',
        'ipykernel',
        'opencv-python',
        'Pillow',
        'tensorflow',
        'Keras',
        'scikit-learn',
        'tqdm',
        'awscli'
    ]
)
