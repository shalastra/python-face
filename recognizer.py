#!/usr/bin/env python

import logging

import cv2
from helper.common import *
from helper.video import *

import sys
sys.path.append("lib/")

from facerec.model import PredictableModel
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.validation import KFoldCrossValidation 
from facerec.serialization import save_model, load_model

from facedet.detector import CascadedDetector

class ExtendedPredictableModel(PredictableModel):

    def __init__(self, feature, classifier, image_size, subject_names):
        PredictableModel.__init__(self, feature=feature, classifier=classifier)
        self.image_size = image_size
        self.subject_names = subject_names

def get_model(image_size, subject_names):
    # Zdefiniwanie metody Fisherfaces jak Extraction Method
    feature = Fisherfaces()
    # Zdefiniowanie 1-NN classifier z EuclideanDistance
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    return ExtendedPredictableModel(feature=feature, classifier=classifier, image_size=image_size, subject_names=subject_names)

def read_subject_names(path):
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
    return folder_names

def read_images(path, image_size=None):
    c = 0
    X = []
    y = []
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if (image_size is not None):
                        im = cv2.resize(im, image_size)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y,folder_names]


class App(object):
    def __init__(self, model, camera_id, cascade_filename):
        self.model = model
        self.detector = CascadedDetector(cascade_fn=cascade_filename, minNeighbors=5, scaleFactor=1.1)
        self.cam = create_capture(camera_id)
            
    def run(self):
        while True:
            ret, frame = self.cam.read()
            # zmiana rozmiaru by przyspieszyc dzialanie
            img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
            imgout = img.copy()
            for i,r in enumerate(self.detector.detect(img)):
                x0,y0,x1,y1 = r

                face = img[y0:y1, x0:x1]
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, self.model.image_size, interpolation = cv2.INTER_CUBIC)

                prediction = self.model.predict(face)[0]
                # Zaznaczenie twarzy
                cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
                # Dopisanie przewidywanej nazwy (bazujac na nazwie folderu)
                draw_str(imgout, (x0-20,y0-20), self.model.subject_names[prediction])
            cv2.imshow('python-face', imgout)

            ch = cv2.waitKey(10)
            if ch == 27:
                break

if __name__ == '__main__':
    from optparse import OptionParser

    size = "100x100"
    cascade_filename = "cascades/haarcascade_frontalface_alt2.xml"

    parser = OptionParser(usage="")
    parser.add_option("-t", action="store", dest="dataset", type="string", default=None,
        help="dataset")

    parser.print_help()
    print "Press [ESC] to exit the program!"

    (options, args) = parser.parse_args()

    if len(args) == 0:
        print "[Error] No prediction model was given."
        sys.exit()

    model_filename = args[0]

    if (options.dataset is None) and (not os.path.exists(model_filename)):
        print "[Error] No prediction model found at '%s'." % model_filename
        sys.exit()

    if not os.path.exists(cascade_filename):
        print "[Error] No Cascade File found at '%s'." % cascade_filename
        sys.exit()
    
    try:
        image_size = (int(size.split("x")[0]), int(size.split("x")[1]))
    except:
        sys.exit()

    if options.dataset:
        if not os.path.exists(options.dataset):
            print "[Error] No dataset found at '%s'." % dataset_path
            sys.exit()    
        
        print "Loading dataset..."
        [images, labels, subject_names] = read_images(options.dataset, image_size)
        
        list_of_labels = list(xrange(max(labels)+1))
        subject_dictionary = dict(zip(list_of_labels, subject_names))
        
        model = get_model(image_size=image_size, subject_names=subject_dictionary)
        
        print "Computing the model..."
        model.compute(images, labels)
        
        print "Saving the model..."
        save_model(model_filename, model)
    # else:
    #     print "Loading the model..."
    #     model = load_model(model_filename)
    
    if not isinstance(model, ExtendedPredictableModel):
        sys.exit()
    
    print "Starting application..."
    App(model=model,
        camera_id=0,
        cascade_filename=cascade_filename).run()
