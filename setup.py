from setuptools import setup, find_packages

setup(
    name='card-classifier',
    version='1.0',
    description='Quiz Backend for playing card classification',
    author='Scott P. White',
    author_email='spwhite1337@gmail.com',
    packages=find_packages(),
    entry_points={'console_scripts': [
        'cc_predictions = card_classifier.api:api_cli',
    ]},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'requests',
        'ipykernel',
        'tqdm',
        'awscli'
    ]
)
