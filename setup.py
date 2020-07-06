from setuptools import setup, find_packages

setup(
    name='card-classifier',
    version='1.0',
    description='Quiz Backend for playing card classification',
    author='Scott P. White',
    author_email='spwhite1337@gmail.com',
    packages=find_packages(),
    entry_points={'console_scripts': [
        'cc_download_magic = scripts.download:download_magic',
        'cc_curate_images = scripts.curate:curate_images',
        'cc_count_cards = scripts.curate:count_cards',
        'cc_run_experiments = scripts.experiments:run_experiments',
        'cc_predictions = scripts.predictions:predict'
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
        'tensorflow',
        'Keras',
        'scikit-learn',
        'tqdm'
    ]
)
