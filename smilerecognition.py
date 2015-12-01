# http://flothesof.github.io/smile-recognition.html
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

faces = datasets.fetch_olivetti_faces()

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


# THIS FUNCTION TAKES THE FRAME FROM THE CAMERA, CONVERTS IT TO GRAYSCALE AND APPLIES THE HAAR CASCADE TO IT
def detect_face(frame):
    cascPath = "C:\Users\ileppane\smilerecognition\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
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
def predict_face_is_smiling(extracted_face):
    return svc_1.predict(extracted_face.ravel())

# THE MAIN PROGRAM
# http://codeplasma.com/2012/12/03/getting-webcam-images-with-python-and-opencv-2-for-real-this-time/
ramp_frames = 30 #Number of frames to throw away while the camera adjusts to light levels
camera = cv2.VideoCapture(0)
def get_image():
    retval, im = camera.read()
    return im
for i in xrange(ramp_frames):
    temp = get_image()
print("Taking image...")
camera_capture = get_image()
del(camera)

input_face = camera_capture
gray, detface = detect_face(input_face)
#face_index = 0
for face in detface:
    (x, y, w, h) = face
    if w > 100:
        extracted_face = extract_face_features(gray, face, (0.15, 0.2)) #(horiz,vert) (0.1, 0.05)=> toimii!
                                                                        #(0.03, 0.05) (0.075, 0.05)
                                                                        # kts. extract_test.py kalibroimisesta
        prediction_result = predict_face_is_smiling(extracted_face)
        cv2.rectangle(input_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #face_index += 1

if prediction_result == 1:
    cv2.imshow("Smile", input_face)
else:
    cv2.imshow("No smile", input_face)    

cv2.waitKey(0)


