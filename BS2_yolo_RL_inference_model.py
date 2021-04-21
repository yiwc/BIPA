import os
import sys
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
import numpy as np
from BS3_yolo_client import YOLO_CLIENT
import cv2
from stable_baselines import PPO2

from yolov4.mymodel import plot_boxes_cv2
import time
from tools import detect_lines
from tools import draw_line
class inference_rl_model(object):
    def __init__(self,dgx,rl_model_name,taskselect,yolo_class_names):

        print("Info!: This inference_rl_model is only for id001 model, with lstm with 32 cpus and envs.")

        self.class_names=yolo_class_names
        self.taskselect=taskselect
        self.yolo=YOLO_CLIENT(dgx=dgx)

        self.hole_pos=[None,None] # [x,y] cam0 [x,y] cam1
        self.head_pos=[None,None] #  [x,y] cam0 [x,y] cam1
        self.refp_pos=[None,None] #  [x,y] cam0 [x,y] cam1
        self.hole_c=[None,None] # 0 cam0 0 cam1
        self.head_c=[None,None] # 0 cam0 0 cam1
        self.lines_detect=[None,None]

        self.observe= None

        self.load_rl(rl_model_name)

        self.dual_imgs=[None,None] # left image, right image
        self.boxes_dualcamera=[None, None]

        #lstm model
        self.lstm_nenvs=32
        self.lstm_state = None
        self.lstm_done = [False for _ in range(self.lstm_nenvs)]

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

        nenvs = 32
        model = PPO2.load(name)
        state = None
        done = [False for _ in range(nenvs)]
        self.model=model

    def get_observe(self): #-> positions

        return self.observe


        # return all_observes
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
            elif box[6]==classB:
                if head_box is None:
                    head_box=box
        if head_box is not None:
            x=(head_box[0]+head_box[2])/2*2-1
            y=(head_box[1]+head_box[3])/2*2-1
            c=head_box[4]*2-1
            if debug_conf:
                c=np.random.uniform(-1,1,1)[0]
                # c=1
            self.head_pos[cam]=[x,y]
            self.head_c[cam]=c
        else:
            if self.head_c[cam] is not None:
                self.head_c[cam]*=0.9
        if hole_box is not None:
            x=(hole_box[0]+hole_box[2])/2*2-1
            y=(hole_box[1]+hole_box[3])/2*2-1
            c=hole_box[4]*2-1

            if debug_conf:
                c=np.random.uniform(-1,1,1)[0]
                # c=1
            self.hole_pos[cam]=[x,y]
            self.hole_c[cam]=c
        else:
            if self.hole_c[cam] is not None:
                self.hole_c[cam]*=0.9

        # yolo_hole, yolo_head
        #boxes [0 0 0 0 0 0 1]
    def input_imgs_to_boxes(self,imgs):
        """imgs -> boxes of dualeyes"""
        raise NotImplementedError("Deprecated method, try input_imgs")
        boxes_dualcamera=[]
        for cam, img in enumerate(imgs):
            boxes = self.yolo.gen(img)
            self.input_box_observe(boxes, cam)
            boxes_dualcamera.append(boxes)

        self.boxes_dualcamera=boxes_dualcamera
        return self.boxes_dualcamera

    def input_imgs(self,imgs,update_detections=True):
        self.dual_imgs=imgs
        if update_detections:
            self.update_sensors()

    def update_sensors(self):
        # print("info: updating sensor 0/4")
        self.update_0_yolo_detect() # Finished
        # print("info: updating sensor 1/4")
        self.update_1_2points() #  Finished
        # print("info: updating sensor 2/4")
        self.update_2_linedetect() # developing / Finished
        # print("info: updating sensor 3/4")
        self.update_3_3rd_point() # developing / Finished
        # print("info: updating sensor 4/4")

    def update_0_yolo_detect(self):
        imgs=self.dual_imgs
        if imgs[0] is None or imgs[1] is None:
            print("Warn: imgs not updated in the RL inference model")
        else:
            boxes_dualcamera = []
            for cam, img in enumerate(imgs):
                boxes = self.yolo.gen(img)
                self.input_box_observe(boxes, cam)
                boxes_dualcamera.append(boxes)

            self.boxes_dualcamera = boxes_dualcamera
            return self.boxes_dualcamera
    @property
    def all_states(self):
        return self.refp_pos+self.head_pos+self.hole_pos
    def check_all_states_updated(self):
        for i in self.all_states:
            if i is None:
                return False
        return True
    def update_1_2points(self):
        # Done already in the self.input_box_observe
        pass
    def update_2_linedetect(self):

        # if self.lines_detect[0] is None or self.lines_detect[1] is None:
        #     print("warn: update_2_linedetect fail -> lines detect is None")
        #     return False

        for i,img in enumerate(self.dual_imgs):
            lines = detect_lines(img, task_select=self.taskselect)
            # print("Lines->",lines)
            if lines is None:
                # print("lines=> ",lines," img==",i)
                # print("info: update_2_linedetect not detected -> line not detected img{}".format(i))
                continue
            if self.lines_detect[i] is None:
                self.lines_detect[i] = lines
            else:
                self.lines_detect[i]=0.1*lines + 0.9*self.lines_detect[i]
            pass
            # img = draw_line(img.copy(), lines)
            # w=5
            # cv2.imshow("imgs with line",
            #            cv2.cvtColor(cv2.resize(img, (int(960 / w), int(720 / w))), cv2.COLOR_RGB2BGR))
        # print("lines detect=>",self.lines_detect)

        # update self.refp_pos
        pass
    def update_3_3rd_point(self):
        # center_yellow_cam0 = self.
        # calculate ref point 0

        if self.lines_detect[0] is None or \
                self.lines_detect[1] is None:
            print("Warn: Fail in update_3_3rd_point => lines detect is None")
            return

        # if self.lines_detect[0].shape[0]>1 or self.lines_detect[1].shape[0]>1:
        #     line_theta=[self.lines_detect[0].mean(0).squeeze()[1],
        #                 self.lines_detect[1].mean(0).squeeze()[1]]
        #
        # else:
        line_theta=[self.lines_detect[0].squeeze()[1],
                    self.lines_detect[1].squeeze()[1]]
        # print("line detect=>",self.lines_detect)
        # self.refp_pos[0]=[self.head_pos[0][0]-(self.head_pos[0][1]+1)*np.cos(line_theta[0]-3/4*np.pi*2),-1]
        # self.refp_pos[1]=[self.head_pos[1][0]-(self.head_pos[1][1]+1)*np.cos(line_theta[1]-1/2*np.pi),-1]
        self.refp_pos[0]=[self.head_pos[0][0]-(self.head_pos[0][1]+1)/np.tan(line_theta[0]-1/2*np.pi),-1]
        self.refp_pos[1]=[self.head_pos[1][0]-(self.head_pos[1][1]+1)/np.tan(line_theta[1]-1/2*np.pi),-1]
        # print(self.refp_pos,"refpos")
        if self.check_all_states_updated():
            pass
        else:
            print("info: update_3_3rd_point=>"+\
                  "RL observe generation not started,"+\
                   "since not all states are not updated for the first time.")
            return False

        x1, y1 = self.refp_pos[0]
        x2, y2 = self.head_pos[0]
        x3, y3 = self.hole_pos[0]
        x4, y4 = self.refp_pos[1]
        x5, y5 = self.head_pos[1]
        x6, y6 = self.hole_pos[1]
        eps = 1e-5

        vec_a_cam0 = [x1 - x2 + eps, y1 - y2 + eps]
        vec_b_cam0 = [x2 - x3 + eps, y2 - y3 + eps]
        vec_a_cam1 = [x4 - x5 + eps, y4 - y5 + eps]
        vec_b_cam1 = [x5 - x6 + eps, y5 - y6 + eps]
        beta_cam0 = np.arctan2(vec_b_cam0[1], vec_b_cam0[0])
        alpha_cam0 = np.arctan2(vec_a_cam0[1], vec_a_cam0[0])
        beta_cam1 = np.arctan2(vec_b_cam1[1], vec_b_cam1[0])
        alpha_cam1 = np.arctan2(vec_a_cam1[1], vec_a_cam1[0])
        p2 = np.pi * 2
        pi = np.pi
        if alpha_cam0 < 0:
            alpha_cam0 += p2
        if alpha_cam1 < 0:
            alpha_cam1 += p2
        if beta_cam0 < 0:
            beta_cam0 += p2
        if beta_cam1 < 0:
            beta_cam1 += p2

        theta_cam0 = beta_cam0 - alpha_cam0
        theta_cam1 = beta_cam1 - alpha_cam1

        if theta_cam0 < -pi:
            theta_cam0 += p2
        if theta_cam1 < -pi:
            theta_cam1 += p2
        if theta_cam0 > pi:
            theta_cam0 -= p2
        if theta_cam1 > pi:
            theta_cam1 -= p2

        final_observe = [theta_cam0, theta_cam1]
        self.observe=final_observe
        pass
            # pass


    # def _state_cont_None(self):
    def tool_draw_point(self,img,point,color=[0,0,0]):
        """point x y => 0~1"""
        def s(pos):
            return int((pos + 1) / 2 * 128)
        if point is None:
            print("Warn: tool_draw_point Fail => point is None")
            return img
        x, y = s(point[0]), s(point[1])
        img = cv2.rectangle(img, (x, y), (x, y), color, 5)
        return img
    def tool_draw_line(self,img,line):
        img = draw_line(img, line)
        return img
    def tool_draw_box(self,img,box):
        drawed = plot_boxes_cv2(img, box, class_names=self.class_names)
        return drawed

    def get_action(self):
        while True:
            try:
                obs = self.get_observe()
                if obs is None:
                    print("Info: RL_interence_model-> get_action-> get_observe -> is None, will retry in 1 s")
                    time.sleep(1)
                else:
                    break
            except Exception as err:
                print(err)
                print("get_action -> ... Can not get_observe <- yolo, yolo output may be None, wait 1 s=> yolo hasnt detect the target object")
                time.sleep(0.5)

        # vec_obs = np.vstack([obs for _ in range(32)])
        # action, state = self.model.predict(vec_obs, state=state, mask=done)
        # # action = self.model.predict(obs, deterministic=True)
        # action = action.mean(0)
        # action = np.squeeze(action)
        # return action

        vec_obs = np.vstack([obs for _ in range(self.lstm_nenvs)])
        action, self.lstm_state =self.model.predict(vec_obs, state=self.lstm_state, mask=self.lstm_done)
        action = action.mean(0)
        action = np.squeeze(action)
        return action
        # if vec:
        # obs=np.expand_dims(obs,0)
        # env.render()

if __name__=="__main__":
    model=inference_rl_model