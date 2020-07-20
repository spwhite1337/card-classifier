# Card Classifier

Buzzfeed style app that outputs the card color for Magic The Gathering from an input image. 

The experiments will fit classification models for a set of pre-trained models in keras (VGG, ResNet50, and Inception).
The models are each "One vs. All" for each mana class (Green, White, Blue, Black, Red, None). The app makes no 
consideration of the input's likelihood of being a magic card. This approach assumes it already is a magic card, just
of an uncertain mana class (Like a Buzzfeed Quiz that already assumes you're a Harry Potter character).

Note: To predict whether or not an input image is a magic card one could create a classifier with all of these images 
in the positive class and something like COCO or ImageNet images in the negative class. I might do this later.

<img src="/docs/mtg_logo.png" alt="MTG Logo" width="512">

## Procedure

- Optional: Set up environment on Ubuntu 18.04 server (e.g. AWS) with:
    - `wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh` (Download Conda)
    - `bash Anaconda3-2018.12-Linux-x86_64.sh` (Accept defaults in prompts)
    - `source .bashrc` for conda commands
    - `conda create -n card-classifier python=3.5`
    - `tmux new -s card-classifier`
    - `conda activate card-classifier` 
    - `pip install --upgrade pip`
    - `sudo apt-get install libsm6 libxrender1 libfontconfig1` (for `open-cv` compatibility)
- `cd card-classifier`
- `pip install -e .`
- `aws configure` (enter AWS keys in prompt, email me for a pair)
- Optional: 
    - Download datasets and results sets with `cc_download --aws` (add `--windows` if on a windows machine)
    - Skip datasets or results sets with `cc_download --aws --skipdata --skipresults`
    - Add a `--dryrun` flag to outline the job without running it
    

## Data

- Download raw data locally with `cc_download`
    - The images are extracted from this API: https://mtgjson.com/. Consider donating to support open source projects.
- Curate data to crop down to artwork with `cc_curate_images`
- Optional: Run `cc_count_cards` to count the cards in each category

## Train

- Configure experiments in `card_classifier/experiments.json`
- Run `cc_run_experiments --version v1`
- Optional: Run `cc_run_experiments --version v0 --debug` for a minimal working model. Or `--overwrite` to restart 
experiments.

## Diagnostics

- The experiments output a diagnostics report and sample cards to `results/`

## Predictions

- Run `cc_predictions --model_type VGG --version v1 --input_path ./data/card_classifier/cc_samples --display_output` 
to generate sample predictions
- Note: samples images should reside in a subdirectory in the input_path
- Output is a dictionary with keys as filenames and values as a dictionary with keys as color and values as 
probabilities of being in that color independent of other colors.



