# Card Classifier

Buzzfeed style app that outputs the card color for Magic The Gathering from an input image. 

The experiments will fit classification models for a set of pre-trained models in keras (VGG, ResNet50, and Inception).
The models are each "One vs. All" for each mana class (Green, White, Blue, Black, Red, None). The app makes no 
consideration of the input's likelihood of being a magic card. This approach assumes it already is a magic card, just
of an uncertain mana class (Like a Buzzfeed Quiz that already assumes you're a Harry Potter character).

Note: To predict whether or not an input image is a magic card one could create a classifier with all of these images 
in the positive class and something like COCO or ImageNet images in the negative class. I might do this later.

<img src="/docs/mtg_logo.png alt="MTG Logo" width="256">

## Procedure

- `cd card-classifier`
- `pip install -e .`

## Data

- Download raw data locally with `cc_download_magic`
    - The images are extracted from this API: https://mtgjson.com/. Consider donating to support open source projects.
- Curate data to crop down to artwork with `cc_curate_images`
- Optional: Run `cc_count_cards` to count the cards in each category

## Train

- Configure experiments in `card_classifier/experiments.json`
- Run `cc_run_experiments`
- Optional: Run `cc_run_experiments --debug` for a minimal working model. 

## Diagnostics

- The experiments output a diagnostics report and sample cards to `results/`

## Predictions

- Run `cc_predictions --model_type VGG --version v1 --input_path ./samples` to generate sample predictions
- Note: samples images should reside in a subdirectory in the input_path
- Output is a dictionary with keys as filenames and values as a dictionary with keys as color and values as predictions



