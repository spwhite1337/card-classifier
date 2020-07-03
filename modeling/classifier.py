import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from keras.applications.vgg16 import VGG16

from sklearn.metrics import auc, roc_curve


class MagicCardClassifier(object):

    def __init__(self,
                 model_type: str = 'VGG'
                 ):
        pass
