from PoseModule import *

cap = cv2.VideoCapture(0)
ptime = 0
detector = poseDetector()

while True:
    success, img = cap.read()
    detector.findPose(img)
    lmList = detector.getPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[0])
        cv2.circle(img, (lmList[0][1], lmList[0][2]),
               15, (0, 0, 255), cv2.FILLED)
    ctime = time.time()

    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (78, 58),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
