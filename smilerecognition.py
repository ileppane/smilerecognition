#!/usr/bin/env python -W ignore::DeprecationWarning
#!/usr/bin/python
# http://flothesof.github.io/smile-recognition.html

# DON'T USE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import os
from sklearn import datasets
from pylab import *
from sklearn import metrics
import json
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
import cv2
from scipy.ndimage import zoom

# THIS FUNCTION TAKES THE FRAME FROM THE CAMERA, CONVERTS IT TO GRAYSCALE AND APPLIES THE HAAR CASCADE TO IT
def detect_face(frame):
    cascPath = "C:\Users\ileppane\smilerecognition\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
    return gray, detected_faces

# this extracts features from the detected face
def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = offset_coefficients[0] * w
    vertical_offset = offset_coefficients[1] * h
    extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
    # zoom to match size of the olivetti faces
    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0], 64. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face

# this predicts smile, i.e. uses the SVC classifier
def predict_face_is_smiling(svc, extracted_face):
    return svc.predict(extracted_face.ravel())
