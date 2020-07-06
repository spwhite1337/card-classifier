# Card Classifier

Buzzfeed style app that outputs the card color for Magic The Gathering from an
input images. 

# Procedure

- `cd card-classifier`
- `pip install -e .`
- [Download data]

# Data

- Download raw data locally with `cc_download_magic`
- Curate data to crop down to artwork with `cc_curate_images`
- Optional: Run `cc_count_cards` to count the cards in each category

# Train

- Configure experiments in `scripts/experiments.json`
- Run `cc_run_experiments`
- Optional: Run `cc_run_experiments --debug` for a minimal working model. 

# Diagnostics

- The experiments output a diagnostics report and sample cards to `modeling/results/`

# Predictions

- Run `cc_predictions --model_type VGG --version v0 --input_path ./samples` to generate sample predictions
- Note: samples images should reside in a subdirectory of in the input_path
- Output is a dictionary with keys as filenames and values as a dictionary with keys as color and values as predictions



