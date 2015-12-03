#!/usr/bin/python
import cv2
from ipywidgets import widgets
from IPython.display import display, clear_output
import smilerecognition
import numpy as np
from random import randint
from sklearn import datasets
from pylab import *
import json
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold

# this is the training dataset
faces = datasets.fetch_olivetti_faces()

# this is the benchmark dataset

####################################################################
# TRAIN THE SVC
# read the existing file where results have been saved through the GUI
class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0
        
    def increment_face(self):
        if self.index + 1 >= len(self.imgs):
            return self.index
        else:
            while str(self.index) in self.results:
                print self.index
                self.index += 1
            return self.index
    
    def record_result(self, smile=True):
        self.results[str(self.index)] = smile

trainer = Trainer()
trainer.results = json.load(open('results.xml'))
svc_1 = SVC(kernel='linear') # initialize
indices = [i for i in trainer.results]
data = faces.data[map(int,indices), :] # image data MUOKATTU: map
target = [trainer.results[i] for i in trainer.results]
target = array(target).astype(int32) # target vector
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)
smilerecognition.train_and_evaluate(smilerecognition.svc_1, X_train, X_test, y_train, y_test)
#
############################################################################

############################################################################
# EVALUATE THE SVC AGAINS OTHER FACE DATA
#
random_image_button = widgets.Button(description="New image!")

def display_face(face):
    clear_output()
    imshow(face, cmap='gray')
    axis('off')
    
def display_face_and_prediction(b):
    index = randint(0, 400)
    face = faces.images[index]
    #face = cv2.imread('AM08HAS.jpg')
    display_face(face)
    print("This is a smile: {0}".format(smilerecognition.svc_1.predict(faces.data[index, :])==1))

random_image_button.on_click(display_face_and_prediction)
display(random_image_button)
display_face_and_prediction(0)


