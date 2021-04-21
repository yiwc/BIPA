#!/usr/bin/env python
import sys
print(sys.path)
from sensor_msgs.msg import Image
rospth='/opt/ros/kinetic/lib/python2.7/dist-packages'
if rospth in sys.path:
    sys.path.remove(rospth)
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append("yolov4")
import rospy
from std_msgs.msg import String

from sensor_msgs.msg import Image
import numpy as np
import cv2


class ros_imgs_test(object):
    def __init__(self):
        pass
        self.img=np.ones([920,1080,3])
    def show(self):
        while True:
            # print("1")

            cv2.imshow("test", self.img)

            k = cv2.waitKey(1)
            # if k
            # print("2")

    def callback(self,data):
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        cv_image = self._data_2_np(data)
        self.img=cv_image
        print("get img")
        # key = cv2.waitKey(2)


        # @ staticmethod
    def _data_2_np(self,data):
        return np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width,-1)

    def listener(self):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('listener', anonymous=True)

        # rospy.Subscriber("/bri_img2", Image, self.callback)
        # rospy.Subscriber("/movo_camera/qhd/image_color", Image, self.callback)
        rospy.Subscriber("/bri_img0", Image, self.callback)
        # rospy.Subscriber("chatter", String, callback)

        # spin() simply keeps python from exiting until this node is stopped



if __name__ == '__main__':
    tester=ros_imgs_test()
    cv2.namedWindow("test")
    tester.listener()
    tester.show()
    rospy.spin()