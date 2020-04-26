from setuptools import setup, find_packages

setup(
    name='card-classifier',
    version='1.0',
    description='Quiz Backend for playing card classification',
    author='Scott P. White',
    author_email='spwhite1337@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'django',
        'plotly',
        'dash',
        'ipykernel',
        'opencv-python',
        'Keras',
    ]
)
