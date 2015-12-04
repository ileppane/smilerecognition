#!/usr/bin/python
# TRAIN THE SVC
import cv2
import numpy as np
from random import randint
from sklearn import datasets
from pylab import *
import json
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import metrics

# this is the training dataset
faces = datasets.fetch_olivetti_faces()

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
# read the existing results-file where results have been saved through the GUI
trainer = Trainer()
trainer.results = json.load(open('results.xml'))

# smile classifier
svc_1 = SVC(kernel='linear') # initialize
indices = [i for i in trainer.results]
data = faces.data[indices, :] # image data
target = [trainer.results[i] for i in trainer.results]
target = array(target).astype(int32) # target vector

# train the smile classifier
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print (scores)
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(
        np.mean(scores), sem(scores)))
    
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)    
    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)    
    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))

train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)
