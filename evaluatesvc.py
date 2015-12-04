#!/usr/bin/python
import cv2
import numpy as np
from random import randint
from sklearn import datasets
from pylab import *
import json
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

# FUNCTION FOR PREDICTING SMILE FROM INPUT FACE IMAGE USING THE SVC CLASSIFIER
def predictsmile(svc, inputface, printoutput, param):
    # printoutput is 1 if the face is wanted as output
    # param are the stretching coefficients, e.g. (0.15,0.2)
    testface = cv2.imread(inputface)
    nodetect = 0 # is assigned 1 if no face is detected from the input testface
    predictionresult = 0 # is assigned 1 if smile is predicted
    gray, detface = detect_face(testface)
    if len(detface) == 1:
        for face in detface:
            (x, y, w, h) = face
            if w > 100: #w pienenee kun etaisyys kamerasta kasvaa
                extractedface = extract_face_features(gray, face, param)
                predictionresult = int(predict_face_is_smiling(svc, extractedface))
                     
                if printoutput == 1:
                    cv2.rectangle(testface, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # PRINT THE OUTPUT
        if printoutput == 1:
            if predictionresult == 1:
                print "smile"
            else:
                print "no smile"        
            subplot(121)
            imshow(cv2.cvtColor(testface, cv2.COLOR_BGR2GRAY), cmap='gray') # show testface
            subplot(122)
            imshow(extractedface, cmap='gray')                              # show extracted face

    else:
        nodetect = 1
        if printoutput == 1:
            print "Error: no face detected"

    return predictionresult, nodetect


