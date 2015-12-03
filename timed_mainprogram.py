#!/usr/bin/env python -W ignore::DeprecationWarning
#!/usr/bin/python
import cv2
import smilerecognition
import cv2
import smilerecognition
import time
import warnings

warnings.simplefilter('ignore', UserWarning)

ramp_frames = 3 #30 Number of frames to throw away while the camera adjusts to light levels

frameCount = 0
prediction_result = []
while frameCount < 10:
    camera = cv2.VideoCapture(0)
    def get_image():      # ONKO TURHA?????
        retval, im = camera.read()
        return im
    for i in xrange(ramp_frames):
        temp = get_image()
    print("Taking image %s" % frameCount)
    camera_capture = get_image()
    del(camera)

    input_face = camera_capture
    gray, detface = smilerecognition.detect_face(input_face)
    if len(detface)==1:
        #face_index = 0
        for face in detface:
            (x, y, w, h) = face
            if w > 100: #w pienenee kun etaisyys kamerasta kasvaa
                extracted_face = smilerecognition.extract_face_features(gray, face, (0.1, 0.05))
                list.append(prediction_result, smilerecognition.predict_face_is_smiling(extracted_face))
                #face_index += 1

        if prediction_result[frameCount] == 1:
            print("Smile")
        else:
            print("No smile")

    else:
        list.append(prediction_result,[2])
        print("Error: no face detected")

    frameCount += 1
    time.sleep(1)
