import os
import sys
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
import numpy as np
from glob import glob
import pathlib
# from gym.envs.classic_control import rendering
# from gan_client import GAN_CLIENT
import pandas as pd
from BS3_yolo_client import YOLO_CLIENT
import cv2
from stable_baselines3 import PPO
import time
class inference_rl_model(object):
    def __init__(self,dgx,rl_model_name,taskselect):

        self.taskselect=taskselect
        self.yolo=YOLO_CLIENT(dgx=dgx)

        self.hole_pos=[None,None] # [x,y] cam0 [x,y] cam1
        self.head_pos=[None,None] #  [x,y] cam0 [x,y] cam1
        self.hole_c=[None,None] # 0 cam0 0 cam1
        self.head_c=[None,None] # 0 cam0 0 cam1
        self.load_rl(rl_model_name)
    @property
    def _state_dict(self):
        d={
            "hole":self.hole_pos,
            "hole_conf":self.hole_c,
            "head":self.head_pos,
            "head_conf":self.head_c,
        }
        return d
    def load_rl(self,name):
        self.model=PPO.load(name)

    def get_observe(self): #-> positions
        detect_observe = [
            self.hole_pos[0][0],
            self.hole_pos[0][1],
            self.head_pos[0][0],
            self.head_pos[0][1],
            self.hole_pos[1][0],
            self.hole_pos[1][1],
            self.head_pos[1][0],
            self.head_pos[1][1]
        ]
        all_observes=[
            detect_observe[2]-detect_observe[0],
            detect_observe[3]-detect_observe[1],
            detect_observe[6]-detect_observe[4],
            detect_observe[7]-detect_observe[5]
        ]
        return all_observes
    def __getattr__(self, item):
        return eval("self.model.{}".format(item))
    def input_box_observe(self,yolo_boxes,cam,debug_conf=True):
        # if cam ==0:

        hole_box=None
        head_box=None

        # insertion
        # class 0 -> hole
        # class 1 -> bolt_head

        # class 0 -> hole -> bolt_bottole -> new class 2
        # class 1 -> bolt_head -> screwhead -> new class 3
        if self.taskselect=="insert":
            classA=0
            classB=1
        elif self.taskselect=="screw":
            classA=2
            classB=3
        else:
            raise NotImplementedError("ERR: YOLO RL Inference MOdel -> Not Defined Task")

        for box in yolo_boxes:
            confidence=box[4]
            if box[6]==classA: # class 00
                if hole_box is None:
                    hole_box=box
                # elif confidence>hole_box[4]:
                #     hole_box=box
            elif box[6]==classB:
                if head_box is None:
                    head_box=box
                # elif confidence>head_box[4]:
                #     head_box=box
        if head_box is not None:
            x=(head_box[0]+head_box[2])/2*2-1
            y=(head_box[1]+head_box[3])/2*2-1
            c=head_box[4]*2-1
            if debug_conf:
                c=np.random.uniform(-1,1,1)[0]
                # c=1
            self.head_pos[cam]=[x,y]
            self.head_c[cam]=c
            # eval("self.head_pos_cam{}".format(cam))=[x,y]
            # eval("self.head_c_cam{}={}".format(cam,c))
        else:
            if self.head_c[cam] is not None:
                self.head_c[cam]*=0.9
            # self.head_c[cam]*=0.9
            # print("no detect in head box-> confidence decay->",self.head_c[cam])
            # eval("self.head_c_cam{}*=0.9".format(cam))
        if hole_box is not None:
            x=(hole_box[0]+hole_box[2])/2*2-1
            y=(hole_box[1]+hole_box[3])/2*2-1
            c=hole_box[4]*2-1

            if debug_conf:
                c=np.random.uniform(-1,1,1)[0]
                # c=1
            self.hole_pos[cam]=[x,y]
            self.hole_c[cam]=c
            # eval("self.hole_box_cam{}={}".format(cam,[x,y]))
            # eval("self.hole_c_cam{}={}".format(cam,c))
        else:
            if self.hole_c[cam] is not None:
                self.hole_c[cam]*=0.9
            # print("no detect in hole_c box-> confidence decay->",self.hole_c[cam])

            #update hole pos
        # yolo_hole, yolo_head
        #boxes [0 0 0 0 0 0 1]
    def input_imgs_observe(self,imgs):
        boxes_dualcamera=[]
        for cam, img in enumerate(imgs):
            boxes = self.yolo.gen(img)
            self.input_box_observe(boxes, cam)
            boxes_dualcamera.append(boxes)
        return boxes_dualcamera
    # def _state_cont_None(self):

    def get_action(self):
        while True:
            try:
                obs = self.get_observe()
                break
            except Exception as err:
                print(err)
                print("get_action -> ... Can not get_observe <- yolo, yolo output may be None, wait 1 s=> yolo hasnt detect the target object")
                time.sleep(0.5)
        print(self._state_dict)
        print("obs :6",obs[:6])
        print("obs 6:",obs[6:])
        action = self.model.predict(obs, deterministic=True)[0]
        return action

if __name__=="__main__":
    model=inference_rl_model