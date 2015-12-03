#!/usr/bin/env python -W ignore::DeprecationWarning
#!/usr/bin/python
import cv2
import smilerecognition

# THE MAIN PROGRAM
# http://codeplasma.com/2012/12/03/getting-webcam-images-with-python-and-opencv-2-for-real-this-time/
ramp_frames = 3 #30 Number of frames to throw away while the camera adjusts to light levels
camera = cv2.VideoCapture(0)
def get_image():                    # ONKO TURHA?????
    retval, im = camera.read()
    return im
for i in xrange(ramp_frames):
    temp = get_image()
print("Taking image...")
camera_capture = get_image()
del(camera)

input_face = camera_capture

gray, detface = smilerecognition.detect_face(input_face)
# detface jaa tyhjaksi () jos ei tunnista kasvoja

if len(detface)==1:
    #face_index = 0
    for face in detface:
        (x, y, w, h) = face
        if w > 100: #w pienenee kun etaisyys kamerasta kasvaa
            extracted_face = smilerecognition.extract_face_features(gray, face, (0.1, 0.05)) # (horiz,vert)
                                                                                         # (0.1, 0.05) => toimii!
                                                                                         # (0.15, 0.2) => antaa aina smile
                                                                                         # (0.03, 0.05) (0.075, 0.05)
                                                                                         # kts. extract_test.py kalibroimisesta
            prediction_result = smilerecognition.predict_face_is_smiling(extracted_face)
            #cv2.rectangle(input_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #face_index += 1

    if prediction_result == 1:
        print "smile"
        #cv2.imshow("Smile", input_face)
    else:
        print "no smile"
        #cv2.imshow("No smile", input_face)    

    #cv2.waitKey(0)
else:
    print "Error: no face detected"
