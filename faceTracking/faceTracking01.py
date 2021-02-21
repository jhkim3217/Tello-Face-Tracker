# https://www.murtazahassan.com/

from djitellopy import tello
import cv2
import os
import time
import numpy as np

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()
# me.takeoff()

###############################
# send_rc_control(self, left_right_velocity,
#                       forward_backward_velocity,
#                       up_down_velocity,
#                       yaw_velocity)
################################
me.send_rc_control(0, 0, 20, 0)
time.sleep(2.0)

w = 360
h = 240
fbRange = [6200, 6800]

### P.I.D
# Kp – The value for the proportional gain Kp
# Ki – The value for the integral gain Ki
# Kd – The value for the derivative gain Kd
pid = [0.4, 0.4, 0]
pError = 0

print(os.getcwd())
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)


def findFace(img):
    # faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

    if img is not None:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    else:
        print("empty frame")
        exit(0)

    myFaceListC = []
    myFaceListArea = []

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x, y), (x + w, y + h), (0,0,255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 255), cv2.FILLED)
        myFaceListC.append(([cx, cy]))
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0,0], 0]


def trackFace(info, w, pid, pError):
    area = info[1]
    x, y = info[0]   # center of area
    fb = 0

    error = x - w//2
    speed = pid[0]*error + pid[1]*(error - pError)
    speed = int(np.clip(speed, -100, 100))

    if fbRange[0] < area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    # elif area == 0:
    #     me.rotate_clockwise(10)

    if x == 0:
        speed = 0
        error = 0
        # me.rotate_clockwise(20)

    print(error, fb)

    ###############################
    # send_rc_control(self, left_right_velocity,
    #                       forward_backward_velocity,
    #                       up_down_velocity,
    #                       yaw_velocity)
    ################################
    me.send_rc_control(0, fb, 0, speed)
    return  error

# cap = cv2.VideoCapture(0)
while True:
    img = me.get_frame_read().frame
    # success, img = cap.read()
    img = cv2.resize(img, (w, h))

    img, info = findFace(img)

    pError = trackFace(info, w, pid, pError)
    print("Area", info[1], "Center", info[0])

    cv2.imshow("Output", img)

    key = cv2.waitKey(1) & 0xff
    if key == ord('l'):
        me.takeoff()
    if key == ord('u'):
        me.send_rc_control(0, 0, 20, 0)
    if key == ord('q'):
        me.land()
        break

# me.land()

# while cap.isOpened():
#     # 카메라 프레임 읽기
#     success, frame = cap.read()
#     if success:
#         # 프레임 출력
#         cv2.imshow('Camera Window', frame)

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     success, img = cap.read()
#     if success:
#         img = cv2.resize(img, (w, h))
#         img, info = findFace(img)
#         pError = trackFace(info, w, pid, pError)
#         print("Area", info[1], "Center", info[0])ㅂ
#
#         cv2.imshow("Output", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             me.land()
#             break
