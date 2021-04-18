from cv2 import cv2
import mediapipe as mp 
import numpy as np 
import time
import math



class poseDetector():
    def __init__(self, mode=False, upBody = False, smooth = True,
                 detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode 
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose =  mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                    self.detectionCon, self.trackCon)

    def findPose(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
    #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS)
        return img
    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                #print(id,lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy),5, (255,8,8), cv2.FILLED)
        return lmList
    



def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = poseDetector()
    
    while True:
        success, img = cap.read()
        detector.findPose(img)
        lmList = detector.getPosition(img,draw=False)
        print(lmList[0])
        cv2.circle(img, (lmList[0][1],lmList[0][2]),15, (0,0,255), cv2.FILLED)
        ctime = time.time()

        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)),(78,58), cv2.FONT_HERSHEY_PLAIN, 3,(255,8,8),3)
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()