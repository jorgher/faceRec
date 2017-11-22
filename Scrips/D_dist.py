# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import os
import csv
import time
import glob
import dlib
import cv2

def angulo(h,w,t):
    uVec = (w[0]-h[0],w[1]-h[1])
    vVec = (t[0]-h[0],t[1]-h[1])
    dot = np.dot(np.array(uVec), np.array(vVec))
    angle = np.arccos(dot/(np.linalg.norm(uVec)*np.linalg.norm(vVec)))
    return angle

# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--path", type=str, default="",
	help="path to input image file")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(rebStart, rebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(lebStart, lebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jwStart, jwEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Captura Base de datos.
fd = open('Emo_2A.csv','wt')
writer = csv.writer(fd)
writer.writerow(("file","EsqIzq","Arriba","EsqDer","Abajo","D1","D2","D3","D4","D5","D6","D7","A0","A1","A2","A3","A4","A5"))

# load the input image, resize it, and convert it to grayscale
for f in glob.glob(os.path.join(args["path"], "*.jpg")):
    print("Processing file: {}".format(f))
    image = cv2.imread(f)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    print("Number of faces detected: {}".format(len(rects)))
    for k, d in enumerate(rects):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array  
        shape = predictor(gray, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        shape = face_utils.shape_to_np(shape)
        # extract landmark coordinates.
        rightEB = shape[rebStart:rebEnd]
        leftEB = shape[lebStart:lebEnd]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        nose = shape[nStart:nEnd]
    
        D1 = dist.euclidean(rightEB[4], leftEB[0])
        D2 = dist.euclidean(rightEB[2], rightEye[1])
        D3 = dist.euclidean(leftEB[2], leftEye[1])
        D4 = dist.euclidean(rightEye[1], rightEye[5])
        D5 = dist.euclidean(leftEye[2],leftEye[4])
        D6 = dist.euclidean(mouth[3],mouth[9])
        D7 = dist.euclidean(mouth[0],mouth[6])

        #print(f,D1,D2,D3,D4,D5,D6,D7)

        A0 = angulo(mouth[0], nose[6], mouth[9])
        A1 = angulo(mouth[9], rightEye[1], rightEB[4])
        A2 = max(angulo(rightEye[1], rightEB[2], rightEB[4]), angulo(leftEye[2], leftEB[0], leftEB[2]))
        A3 = angulo(nose[6], rightEB[4], leftEB[0])
        A4 = angulo(mouth[0], mouth[9], mouth[3])
        A5 = max(angulo(mouth[0], mouth[9], nose[6]),angulo(mouth[6],mouth[9],nose[6]))
 
        #print(A0,A1,A2,A3,A4,A5)

        writer.writerow((f,d.left(),d.top(),d.right(),d.bottom(),D1,D2,D3,D4,D5,D6,D7,A0,A1,A2,A3,A4,A5))

fd.close()



