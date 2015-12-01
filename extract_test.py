import cv2
from matplotlib.patches import Rectangle
from pylab import *
from scipy.ndimage import zoom
from IPython.display import display, clear_output
#%matplotlib inline

input_face = cv2.imread('test.jpg')
cascPath = "C:\Users\ileppane\smilerecognition\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
gray = cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY)
detected_faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(100, 100),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
detected_faces
ax = gca()
ax.imshow(gray, cmap='gray')
for (x, y, w, h) in detected_faces:
    ax.add_artist(Rectangle((x, y), w, h, fill=False, lw=5, color='blue'))

original_extracted_face = gray[y:y+h, x:x+w]
horizontal_offset = 0.15 * w # 0.15
vertical_offset = 0.2 * h    # 0.2
extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]

subplot(121)
imshow(original_extracted_face, cmap='gray')
subplot(122)
imshow(extracted_face, cmap='gray')

new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0], 64. / extracted_face.shape[1]))

new_extracted_face = new_extracted_face.astype(float32)

new_extracted_face /= float(new_extracted_face.max())

def display_face(face):
    clear_output()
    imshow(face, cmap='gray')
    axis('off')

display_face(new_extracted_face[:, :])

