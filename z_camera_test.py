import os
import sys
rospth1='/opt/ros/kinetic/lib/python2.7/dist-packages'
rospth2='/home/radoe-1/movo_ws/devel/lib/python2.7/dist-packages'
# # rospth='/opt/ros/kinetic/lib/python2.7/dist-packages'
# if rospth in sys.path:
sys.path.remove(rospth1)
sys.path.remove(rospth2)

import cv2
import time
import numpy as np
allow_save=True

def img_process(img):
    img = cv2.resize(img, (320, 240))
    c_ = [int(img.shape[0] / 2), int(img.shape[1] / 2), 3]
    size = 65
    img = img[c_[0] - size:c_[0] + size, c_[1] - size:c_[1] + size, :]
    img = np.array(img).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    return img

def img_process_zoom_hole(img):
    img = cv2.resize(img, (480*2, 640*2))
    c_ = [int(img.shape[0] *0.2), int(img.shape[1] * 0.5), 3]

    size = np.random.randint(40,64)
    img = img[c_[0] - size:c_[0] + size, c_[1] - size:c_[1] + size, :]
    img = np.array(img).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    return img

camid=[1,2]

cam = cv2.VideoCapture(camid[0])
cam.set(4,160)
cam.set(3,120)

twocamera=len(camid)==2
if twocamera:
    cam2 = cv2.VideoCapture(camid[1])
    cam2.set(4,160)
    cam2.set(3,120)



cv2.namedWindow("test")
cv2.namedWindow("test2")
img_counter = 0

dirname=str(int(time.time()))
dirpth=os.path.join(os.getcwd(),"saveimgs",dirname)
if allow_save : os.makedirs(dirpth)

i=0
while True:
    ret, frame = cam.read()
    if twocamera:
        ret2, frame2 = cam2.read()

    # frame=img_process_zoom_hole(frame)
    try:
        frame=img_process(frame)
        if twocamera:
            frame2=img_process(frame2)
    except:
        continue
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    if twocamera:
        cv2.imshow("test2", frame2)


    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        pass
    i+=1
    if i % 10==0 and allow_save:
        img_name = dirpth+"/frame_t{}_{}.png".format(dirname,img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        if twocamera:
            img_name = dirpth+"/frame_t{}_{}.png".format(dirname,img_counter)
            cv2.imwrite(img_name, frame2)
            print("{} written!".format(img_name))
            img_counter += 1
    time.sleep(0.1)
cam.release()

cv2.destroyAllWindows()