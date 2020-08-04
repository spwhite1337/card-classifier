# Card Classifier (Minified)

Buzzfeed style app that outputs the card color for Magic The Gathering from an input image. 

The experiments fit classification models with pre-trained models in keras (VGG16). The models are each "One vs. All" 
for each mana class (Green, White, Blue, Black, Red, None). The app makes no consideration of the input's likelihood of 
being a magic card. This approach assumes it already is a magic card, just of an uncertain mana class (Like a Buzzfeed 
Quiz that already assumes you're a Harry Potter character).

Note: To predict whether or not an input image is a magic card one could create a classifier with all of these images 
in the positive class and something like COCO or ImageNet images in the negative class. I might do this later.

<img src="/docs/mtg_logo.png" alt="MTG Logo" width="512">

---
<p>
    <img src="docs/keras_logo.png" alt="Keras Logo" width="256">
</p>

---

## Procedure

- `cd card-classifier`
- `pip install -e .`

## Predictions

This minified version just loads a dictionary of results from previously input sample images. It does not
work for new images. It is intended for lightweight development of the Website that serves it.

- Run `cc_predictions --version v1 --input_path ./data/card_classifier/cc_samples --display_output` 
to output "results"
```
2020-07-25 11:34:41 INFO     Output:
{   'images\\balrog.jpg': {   'B': 0.9783472,
                              'G': 0.045847435,
                              'N': 0.98153555,
                              'R': 0.049625058,
                              'U': 0.0010573716,
                              'W': 0.8914076},
    'images\\galadriel.jpg': {   'B': 0.7733727,
                                 'G': 0.26406303,
                                 'N': 0.05677448,
                                 'R': 0.32482594,
                                 'U': 0.0024097085,
                                 'W': 0.03252216},
    'images\\javert.jpg': {   'B': 6.172084e-10,
                              'G': 0.15237981,
                              'N': 0.36429337,
                              'R': 0.010537634,
                              'U': 0.019179326,
                              'W': 0.03924815},
    'images\\jean.jpg': {   'B': 0.62263674,
                            'G': 0.13248377,
                            'N': 0.93159884,
                            'R': 0.056817085,
                            'U': 0.009865508,
                            'W': 0.99986947},
    'images\\link.jpg': {   'B': 0.14316888,
                            'G': 0.49023804,
                            'N': 0.0003216137,
                            'R': 0.12879494,
                            'U': 0.0009454357,
                            'W': 0.024071183},
    'images\\mary.jpg': {   'B': 0.5267415,
                            'G': 0.10777836,
                            'N': 0.051620048,
                            'R': 0.0004039896,
                            'U': 0.008130033,
                            'W': 0.010184212},
    'images\\napolean.jpg': {   'B': 0.015787182,
                                'G': 0.107250236,
                                'N': 4.5033044e-09,
                                'R': 0.014052201,
                                'U': 0.0012653382,
                                'W': 0.024982987},
    'images\\sauron.jpg': {   'B': 0.45304516,
                              'G': 1.6971251e-05,
                              'N': 0.097894326,
                              'R': 0.08809588,
                              'U': 0.0010831527,
                              'W': 0.06654953},
    'images\\tolstoy.jpg': {   'B': 0.30917725,
                               'G': 0.14649346,
                               'N': 0.39106447,
                               'R': 0.27552798,
                               'U': 0.0029577622,
                               'W': 0.063730784},
    'images\\vader.jpeg': {   'B': 0.3224331,
                              'G': 0.40748483,
                              'N': 0.25855398,
                              'R': 0.017555842,
                              'U': 0.035545412,
                              'W': 0.025173346}}
```

