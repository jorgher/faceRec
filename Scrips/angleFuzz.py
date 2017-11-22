# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import pandas as pd
import argparse
import imutils
import os
import csv
import time
import glob
import dlib
import cv2

def g(x):
    if x == A0:
        


   
def f(xs,xmm,xm,xmp,xl):
    xs = XS
    xmm = XMM
    xm = XM
    xmp = XMP
    xl = XL
    if x >= XS and x < XMM:
        y_ = lambda x : 1.0 - 1.0/(XM-XS)*(x - XS)
        y  = y_(x)
        fz = "S"        
    elif x < XM:
        y1 = lambda x : 1.0 - 1.0/(XM-XS)*(x - XS)
        y2 = lambda x : (1/(XM-XMM))*(x - XMM)
        y = max(y1(x),y2(x))
        if y == y1(x):
            fz = "S" 
        elif y == y2(x):
            fz = "M"
    elif x < XMP:
        y3 = lambda x : 1 + (-1/(XMP-XM))*(x - XM)
        y4 = lambda x : (1/(XL-XMP))*(x - XMP)
        y = max(y3(x),y4(x))
        if y == y3(x):
            fz = "M" 
        elif y == y4(x):
            fz = "L"
    elif x <= XL:
        y_ = lambda x : (1/(XL-XM))*(x - XM)
        y = y_(x)
        fz = "L"
    else:
        y = "El valor no se encuentra en el dominio"
    return y, fz

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
ap.add_argument("-i", "--path", type=str, default="",
	help="path to input image file")
ap.add_argument("-d", "--memb-function", type=str, default="",
	help="path to membership function")
args = vars(ap.parse_args())

print("Loading membership function coordinates"
file = pd.read_csv(args["memb-function"])
db = pd.DataFrame(file)
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

image = cv2.imread(argparse["path"]
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)
print("Number of faces detected: {}".format(len(rects)))
for k, d in enumerate(rects):
    shape = predictor(gray, d)
    shape = face_utils.shape_to_np(shape)
    
    # extract landmark coordinates.
    rightEB = shape[rebStart:rebEnd]
    leftEB = shape[lebStart:lebEnd]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth = shape[mStart:mEnd]
    nose = shape[nStart:nEnd]
        
    # Calulo de angulos
    A0 = angulo(mouth[0], nose[6], mouth[9])
    A1 = angulo(mouth[9], rightEye[1], rightEB[4])
    A2 = max(angulo(rightEye[1], rightEB[2], rightEB[4]), angulo(leftEye[2], leftEB[0], leftEB[2]))
    A3 = angulo(nose[6], rightEB[4], leftEB[0])
    A4 = angulo(mouth[0], mouth[9], mouth[3])
    A5 = max(angulo(mouth[0], mouth[9], nose[6]),angulo(mouth[6],mouth[9],nose[6]))

    #Tuplas para cada angulo.
    
    



