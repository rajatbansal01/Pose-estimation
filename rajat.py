from cv2 import cv2
import mediapipe as mp 
import numpy as np 
import time

mpDraw = mp.solutions.drawing_utils
mpPose =  mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)
ptime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,
                              mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id,lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx,cy),5, (255,8,8), cv2.FILLED)

    
    ctime = time.time()

    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)),(78,58), cv2.FONT_HERSHEY_PLAIN, 3,(255,8,8),3)
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

