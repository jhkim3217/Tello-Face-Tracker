# from djitellopy import tello
from time import sleep
import cv2
# me = tello.Tello()
# me.connect()
# print(me.get_battery())
#
# me.streamon()
h = 360 * 2
w = 240 * 2

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    img = cv2.resize(img, (h, w))
    # img, info = findFace(img)
    # pError = trackFace(info, w, pid, pError)
    # print("Area", info[1], "Center", info[0])
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # me.land()
        break
