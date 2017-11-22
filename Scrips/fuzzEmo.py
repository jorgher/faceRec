# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import pandas as pd
import numpy as np
import csv
import datetime
import argparse
import imutils
import time
import dlib
import cv2
 
def angulo(h,w,t):
    uVec = (w[0]-h[0],w[1]-h[1])
    vVec = (t[0]-h[0],t[1]-h[1])
    dot = np.dot(np.array(uVec), np.array(vVec))
    angle = np.arccos(dot/(np.linalg.norm(uVec)*np.linalg.norm(vVec)))
    return angle

def f(x,XS,XMM,XM,XMP,XL):
    if x > 0 and x < XS:
        y = 1 #"El valor no se encuentra en el dominio"
        fz = "S"
    if x >= XS and x <= XMM:
        y_ = lambda x : 1.0 - 1.0/(XM-XS)*(x - XS)
        y  = y_(x)
        fz = "S"        
    elif x <= XM:
        y1 = lambda x : 1.0 - 1.0/(XM-XS)*(x - XS)
        y2 = lambda x : (1/(XM-XMM))*(x - XMM)
        y = max(y1(x),y2(x))
        if y == y1(x):
            fz = "S" 
        elif y == y2(x):
            fz = "M"
    elif x <= XMP:
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
        y = 1 #"El valor no se encuentra en el dominio"
        fz = "L"
    return y, fz

def coord(Emo,A0,A1,A2,A3,A4,A5):
    file = pd.read_csv("membeshipFunct_B.csv")
    index = ['Emotion','Angl']
    file_ = file.set_index(index)
    cord = []
    Angls = ['A0','A1','A2','A3','A4','A5']
    #Emotions = ['Emo_1']
    for Ang in Angls:
        valCord = file_.loc[Emo].loc[Ang]
        cord.append(valCord)
            
    return cord 

def fuzzReg(re):
    A0 = re[0]
    A1 = re[1]
    A2 = re[2]
    A3 = re[3]
    A4 = re[4]
    A5 = re[5]
    Ie = ""
    Em = ""
    if A0 == 'L' and A1 == 'M' and A2 == 'M' and A3 == 'M' and A4 == 'S' and A5 == 'L':
        Ie == 'St'
        Em == 'E2'
    if A0 == 'L' and A1 == 'M' and A2 == 'M' and A3 == 'M' and A4 == 'M' and A5 == 'L':
        Ie == 'St'
        Em == 'E2'
    if A0 == 'L' and A1 == 'M' and A2 == 'M' and A3 == 'M' and A4 == 'M' and A5 == 'M':
        Ie == 'We'
        Em == 'E2'
    if A0 == 'L' and A1 == 'M' and A2 == 'M' and A3 == 'M' and A4 == 'S' and A5 == 'M':
        Ie == 'We'
        Em == 'E2'
    if A0 == 'M' and A1 == 'M' and A2 == 'M' and A3 == 'M' and A4 == 'S' and A5 == 'M':
        Ie == 'St'
        Em == 'E3'
    if A0 == 'M' and A1 == 'M' and A2 == 'M' and A3 == 'M' and A4 == 'S' and A5 == 'S':
        Ie == 'St'
        Em == 'E3'
    if A0 == 'M' and A1 == 'L' and A2 == 'M' and A3 == 'M' and A4 == 'S' and A5 == 'M':
        Ie == 'We'
        Em == 'E3'
    if A0 == 'M' and A1 == 'L' and A2 == 'M' and A3 == 'M' and A4 == 'M' and A5 == 'S':
        Ie == 'We'
        Em == 'E3'
    if A0 == 'S' and A1 == 'M' and A2 == 'S' and A3 == 'M' and A4 == 'L' and A5 == 'L':
        Ie == 'St'
        Em == 'E5'
    if A0 == 'S' and A1 == 'S' and A2 == 'S' and A3 == 'M' and A4 == 'L' and A5 == 'L':
        Ie == 'St'
        Em == 'E5'
    if A0 == 'S' and A1 == 'M' and A2 == 'S' and A3 == 'L' and A4 == 'L' and A5 == 'L':
        Ie == 'We'
        Em == 'E5'
    if A0 == 'S' and A1 == 'S' and A2 == 'S' and A3 == 'L' and A4 == 'L' and A5 == 'L':
        Ie == 'We'
        Em == 'E5'
    if A0 == 'M' and A1 == 'M' and A2 == 'M' and A3 == 'S' and A4 == 'S' and A5 == 'S':
        Ie == 'St'
        Em == 'E4'
    if A0 == 'M' and A1 == 'M' and A2 == 'M' and A3 == 'S' and A4 == 'S' and A5 == 'M':
        Ie == 'St'
        Em == 'E4'
    if A0 == 'M' and A1 == 'M' and A2 == 'L' and A3 == 'S' and A4 == 'S' and A5 == 'M':
        Ie == 'We'
        Em == 'E4'
    if A0 == 'M' and A1 == 'M' and A2 == 'L' and A3 == 'S' and A4 == 'S' and A5 == 'S':
        Ie == 'We'
        Em == 'E4'
    
    return Ie, Em

#def defuzz(Ie):
    
  
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

fd = open('reglasC.csv','wt')
writer = csv.writer(fd)
#writer.writerow(("xs","xmm","xm","xmp","xl","Emotion","Angl","ValAng","FuzzValue","FuzzStr","rule"))
writer.writerow(("A0","A1","A2","A3","A4","A5,","Emotion","Angl","ValAng","FuzzValue","FuzzStr","rule"))
writer.writerow(("A0","A1","A2","A3","A4","A5,","Emotion","Angl"))
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(rebStart, rebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(lebStart, lebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jwStart, jwEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
        rects = detector(gray, 0)

	# loop over the face detections
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            rightEB = shape[rebStart:rebEnd]
            leftEB = shape[lebStart:lebEnd]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            nose = shape[nStart:nEnd]

            A0 = angulo(mouth[0], nose[6], mouth[9])
            A1 = angulo(mouth[9], rightEye[1], rightEB[4])
            A2 = max(angulo(rightEye[1], rightEB[2], rightEB[4]), angulo(leftEye[2], leftEB[0], leftEB[2]))
            A3 = angulo(nose[6], rightEB[4], leftEB[0])
            A4 = angulo(mouth[0], mouth[9], mouth[3])
            A5 = max(angulo(mouth[0], mouth[9], nose[6]),angulo(mouth[6],mouth[9],nose[6]))
            angls = [A0,A1,A2,A3,A4,A5]
            print(angls)
            Angls = ['A0','A1','A2','A3','A4','A5']
            Emotions = ['Emo_1','Emo_2','Emo_3','Emo_4','Emo_5','Emo_6','Emo_7']
            #Emotions = ['Emo_2','Emo_3','Emo_4','Emo_5']
            reglas = []
            rule = 1.
            for emo in Emotions:
                #print(emo)
                emo_ = emo
                cord = coord(emo,A0,A1,A2,A3,A4,A5)
                for ang,av,data in zip(Angls,angls,cord):
                #for data in cord:
                    xs,xmm,xm,xmp,xl = data
                    #print(xs,xmm,xm,xmp,xl)
                    # Fuzzifica el valor del angulo
             #       for ang,av in zip(Angls,angls):
                        #print(av)
                    k,e = f(av,xs,xmm,xm,xmp,xl)
                    rule = rule*k
                        #print(emo,av,k,e)
                    #writer.writerow((xs,xmm,xm,xmp,xl,emo_,ang,av,k,e,rule))
                    writer.writerow((A0,A1,A2,A3,A4,A5,emo_,ang,av,k,e,rule))
                    reglas.append(e)
                    #defuzz(regla(e))   
            rE1 = reglas[:6]
            rE2 = reglas[6:12]
            rE3 = reglas[12:18]
            rE4 = reglas[18:24]
            rE5 = reglas[24:30]
            rE6 = reglas[30:36]
            rE7 = reglas[36:42]
            reg = [rE1,rE2,rE3,rE4,rE5,rE6,rE7]
            #reg = [rE1,rE2,rE3,rE4]
            
            for i,r in enumerate(reg):
                print(list(r))
                EM, InE = fuzzReg(r)
                print('Intensidad Emosion {}: = {}:'. format(EM,InE))
                writer.writerow((r[0],r[1],r[2],r[3],r[4],r[5],EM,InE))

	  
	# show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
