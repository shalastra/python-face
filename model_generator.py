#!/usr/bin/env python

import os
import sys

sys.path.append("../..")

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

from builtins import range

from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
from facerec.serialization import save_model, load_model

import numpy as np
try:
    from PIL import Image
except ImportError:
    import Image
import logging


def read_images(path, sz=None):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")

                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as e:
                    print("I/O error: {0}".format(e))
                    raise e
                except:
                    print("Unexpected error: {0}".format(sys.exc_info()[0]))
                    raise
            c = c+1
    return [X,y]

if __name__ == "__main__":
    out_dir = None

    if len(sys.argv) < 2:
        sys.exit()
    [X,y] = read_images(sys.argv[1])

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    feature = Fisherfaces()

    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)

    my_model = PredictableModel(feature=feature, classifier=classifier)

    my_model.compute(X, y)

    save_model('model.pkl', my_model)
