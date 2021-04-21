# v3
# input is dual image
# go through rl server

import sys
rospth1='/opt/ros/kinetic/lib/python2.7/dist-packages'
rospth2='/home/radoe-1/movo_ws/devel/lib/python2.7/dist-packages'
# # rospth='/opt/ros/kinetic/lib/python2.7/dist-packages'
# if rospth in sys.path:
sys.path.remove(rospth1)
sys.path.remove(rospth2)
sys.path.append(rospth1)
sys.path.append(rospth2)

import cv2
import time
# from stable_baselines import PPO2
# from ppo2 import PPO2
import numpy as np
import os
try:
    from gym.envs.classic_control import rendering
except:
    pass

from gan_client import GAN_CLIENT
from threading import Thread
from BS3_server_client import RL_CLIENT
import rospy
sys.path.append(rospth1)
sys.path.append(rospth2)
print(sys.path)
IMG_COLLECT_ABLE=False

if __name__=="__main__":

    rospy.init_node('listener', anonymous=True)
    real_arm=True

    # ri=RealInsert()
    stps = 0
    loop_enable = False



    if real_arm:
        from kinova_ct_v2 import arm_control
        arm = arm_control()
        ARM = "right"

        arm.set_gripper(0)
        print("To Reset...")
        arm.reset("insert",hard=False,f=15)
        # arm.move_relate("right",[0.1*100,0.1*100,0.1*100],maxforce=1000)
        # arm.move_relate("left",[0.1*100,0.1*100,0.1*100],maxforce=1000)
        # arm.set_gripper(0)
        # arm.set_gripper(1)
        # arm.set_gripper(0)
