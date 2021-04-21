# v3
# input is dual image
# go through rl server
import math
import sys
import rospy
from squaternion import Quaternion
rospth1='/opt/ros/kinetic/lib/python2.7/dist-packages'
rospth2='/home/radoe-1/movo_ws/devel/lib/python2.7/dist-packages'
# # rospth='/opt/ros/kinetic/lib/python2.7/dist-packages'
# if rospth in sys.path:
sys.path.remove(rospth1)
sys.path.remove(rospth2)

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

sys.path.append(rospth1)
sys.path.append(rospth2)
print(sys.path)
IMG_COLLECT_ABLE=False

if __name__=="__main__":

    real_arm=True
    rospy.init_node("test")
    # ri=RealInsert()
    stps = 0
    loop_enable = False



    if real_arm:
        from kinova_ct_v2 import arm_control
        arm = arm_control()
        ARM = "right"
        print("To Reset...")


        duration = 300
        f=15
        hard=False
        l_max_force = [f for i in range(6)]
        r_max_force = [f for i in range(6)]
        max_force=[f for i in range(6)]
        d = arm.odyssey.gval.default_insert1
        # arm.odyssey._L0_upper_jp_move_safe(
        #     d["l"], d['r'], d['h'], d['lin'], duration, l_max_force, r_max_force, wait=True, hard=hard
        # )

        print(arm.odyssey.get_rpose())
        arm.odyssey.
        # [0.2895526854052859, 0.6434105260436278, 1.272755405839076,
        #  0.6530300346320159, 0.2607317891002266, -0.6615815082166981, 0.2605390872917587]
        #
        # q=Quaternion(0.6530300346320159, 0.2607317891002266, -0.6615815082166981, 0.2605390872917587)
        # euler=q.to_euler(degrees=False)
        # # euler=[-2.78+0.3, -1.55, -2.74]
        # arm.odyssey.arm_cart_move('right',pos=[0.289,0.643,1.272],orn=euler,maxforce=max_force,wait=True,hard=False)
        # print(euler)

        # arm.odyssey.arm_cart_move("right",pos=[0.2895526854052859, 0.6434105260436278, 1.272755405839076],orn=)
        # arm.odyssey.moveta


        # arm.odyssey.single_move_relate()

        # arm.odyssey

        # arm.odyssey._L0_upper_jp_move_safe(
        #     d["l"],d['r'],d['h'],d['lin'],duration,f,f,wait=True,hard=hard
        # )
        # arm.set_gripper(0)
        # arm.set_gripper(1)
        # arm.set_gripper(0)
        # arm.reset(hard=False,f=20)
        # arm.movo_collect_dataset("right")

