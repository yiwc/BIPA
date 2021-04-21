# v3
# input is dual image
# go through rl server

import sys

try:
    rospth = '/opt/ros/kinetic/lib/python2.7/dist-packages'
    if rospth in sys.path:
        sys.path.remove(rospth)
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.append("yolov4")
except:
    pass
import cv2
import time
import numpy as np
import os
try:
    from gym.envs.classic_control import rendering
except:
    pass
import random
from BS3_yolo_RL_inference_model import inference_rl_model
from threading import Thread
from yolov4.mymodel import plot_boxes_cv2

class RealInsert(object):

    def __init__(self,rl_model_name,taskselect):
        self.taskselect=taskselect
        # self.yolo=YOLO_CLIENT()
        self.model=inference_rl_model(dgx=True,rl_model_name=rl_model_name,taskselect=taskselect)
        self.yolo=self.model.yolo
        self.gan_able = False
        self.cam_img_size = CAM_IMG_SIZE # 65 90


        self.save_dir_name=str(time.time()).split('.')[0]
        os.makedirs("saveimgs/dualeyes/"+self.save_dir_name)
        os.makedirs("saveimgs/dualeyes/"+self.save_dir_name+"draw")
        self.img_collect_able= IMG_COLLECT_ABLE
        self.img_counter=0

        self.imgs=[None,None]
        self.image_subs=[]
        self.image_callbacks=[
            self.image_callback1,
            self.image_callback2,
        ]


        if not ENABLE_FAKE_FLOW_TEST:
            rospy.Subscriber("/bri_img0" , Image, self.image_callback1)
            rospy.Subscriber("/bri_img1" , Image, self.image_callback2)
        else:
            self.load_testimgs()
        self.bridge = CvBridge()
        self.cam_devices=[None,None]#cv2.VideoCapture(0),cv2.VideoCapture(2)
        self.imgs_captured=[None,None]
        self.dualimg_captured=None
        self.img_captured = np.random.uniform(0,255,[128,256,3])
        seris_id=[1,2] # camera series id, [cam1,left, cam2, right]
        while ( self.imgs[0] is None or self.imgs[1] is None):
            print("wait the first images to come in...",self.imgs)
            time.sleep(0.5)
        print("all images loaded")

        dual_cam_thread=Thread(target=self.keep_cap_dualimg,args=(seris_id,))
        dual_cam_thread.start()

        self.gym_viewer=rendering.SimpleImageViewer()
        self.gym_viewer2=rendering.SimpleImageViewer()

        print("model loaded")
    @ staticmethod
    def _data_2_np(data):
        return np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width,-1)
    def image_callback1(self,data):
        cv_image = self._data_2_np(data)
        self.imgs[0]=cv_image
        pass
    def image_callback2(self,data):
        cv_image = self._data_2_np(data)
        self.imgs[1]=cv_image
        pass

    def load_testimgs(self):

        def load_imgs(self):
            while True:
                print("load imgs..")
                time.sleep(0.1)
        thr = Thread(target=load_imgs, args=())
        thr.start()

    def keep_cap_dualimg(self,seris_id):

        def img_process(img):
            img = cv2.resize(img, (320, 240))
            c_ = [int(img.shape[0] / 2), int(img.shape[1] / 2), 3]
            size = self.cam_img_size # 65 insert
            img = img[c_[0] - size:c_[0] + size, c_[1] - size:c_[1] + size, :]
            img = np.array(img).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            return img

        while True:
            time.sleep(0.04)
            frame1=self.imgs[0]
            frame2=self.imgs[1]
            if not ENABLE_FAKE_FLOW_TEST:
                frame2=img_process(frame2)
                frame1=img_process(frame1)

            dual_img = cv2.hconcat([frame1, frame2])
            key = cv2.waitKey(2)
            if key == 27:  # exit on ESC
                break

            self.imgs_captured[0]=frame1
            self.imgs_captured[1]=frame2
            drawed = self.imgs_captured.copy()

            boxes_dual=self.model.input_imgs_observe(self.imgs_captured.copy())
            class_names=[
                "hole",
                "bolt_head",
                "bolt_bottom",
                "screw_head"
            ]

            drawed[0]=plot_boxes_cv2(drawed[0], boxes_dual[0], class_names=class_names)
            drawed[1]=plot_boxes_cv2(drawed[1], boxes_dual[1], class_names=class_names)

            try:
                def s(pos):
                    return  int((pos+1)/2*128)

                hole1x,hole1y= s(self.model.hole_pos[0][0]),s(self.model.hole_pos[0][1])
                hole2x,hole2y= s(self.model.hole_pos[1][0]),s(self.model.hole_pos[1][1])
                head1x,head1y= s(self.model.head_pos[0][0]),s(self.model.head_pos[0][1])
                head2x,head2y= s(self.model.head_pos[1][0]),s(self.model.head_pos[1][1])
                drawed[0] = cv2.rectangle(drawed[0], (hole1x, hole1y), (hole1x, hole1y), [0,0,0], 5)
                drawed[0] = cv2.rectangle(drawed[0], (head1x, head1y), (head1x, head1y), [125,0,0], 5)
                drawed[1] = cv2.rectangle(drawed[1], (hole2x, hole2y), (hole2x, hole2y), [0,0,0], 5)
                drawed[1] = cv2.rectangle(drawed[1], (head2x, head2y), (head2x, head2y), [125,0,0], 5)
            except Exception as err:
                print("tring to mark observation point,",err)
                pass
            self.dualimg_captured = dual_img
            self.img_captured=self.dualimg_captured
            w=2

            if(self.img_collect_able):
                thresh=0.2
                if LOG_IN_EXP:
                    thresh=0.8
                if random.random()<thresh:
                    pass
                else:
                    self.img_counter += 1
                    cv2.imwrite("saveimgs/dualeyes/" + self.save_dir_name +"/"+ self.save_dir_name +"_"+ str(self.img_counter) + "_l.png",
                                cv2.cvtColor(self.imgs_captured[0], cv2.COLOR_RGB2BGR))
                    cv2.imwrite("saveimgs/dualeyes/" + self.save_dir_name+"/" + self.save_dir_name +"_"+ str(self.img_counter) + "_r.png",
                                cv2.cvtColor(self.imgs_captured[1], cv2.COLOR_RGB2BGR))
                    cv2.imwrite("saveimgs/dualeyes/" + self.save_dir_name+"/" + self.save_dir_name +"_"+ str(self.img_counter) + "_d.png",
                                cv2.cvtColor(self.dualimg_captured, cv2.COLOR_RGB2BGR))
                    cv2.imwrite("saveimgs/dualeyes/" + self.save_dir_name +"draw/"+ self.save_dir_name +"_"+ str(self.img_counter) + "_l.png",
                                cv2.cvtColor(drawed[0], cv2.COLOR_RGB2BGR))
                    cv2.imwrite("saveimgs/dualeyes/" + self.save_dir_name+"draw/" + self.save_dir_name +"_"+ str(self.img_counter) + "_r.png",
                                cv2.cvtColor(drawed[1], cv2.COLOR_RGB2BGR))

                    print("Collecting Image->",self.img_counter)
            cv2.imshow("Left/Right Eye (BGR)",
                       cv2.cvtColor(cv2.resize(dual_img, (int(960 / w), int(720 / w))), cv2.COLOR_RGB2BGR))
            # cv2.imshow("Left (BGR)",
            #            cv2.cvtColor(cv2.resize(self.imgs_captured[0], (int(360 / w), int(360 / w))), cv2.COLOR_RGB2BGR))
            # cv2.imshow("Right (BGR)",
            #            cv2.cvtColor(cv2.resize(self.imgs_captured[1], (int(360 / w), int(360 / w))), cv2.COLOR_RGB2BGR))
            cv2.imshow("Left (BGR)",
                       cv2.cvtColor(cv2.resize(drawed[0], (int(360 / w), int(360 / w))), cv2.COLOR_RGB2BGR))
            cv2.imshow("Right (BGR)",
                       cv2.cvtColor(cv2.resize(drawed[1], (int(360 / w), int(360 / w))), cv2.COLOR_RGB2BGR))

    def switch_yolo(self):
        self.img_show_yolo=1-self.img_show_yolo
    def get_action(self):
        return self.model.get_action()

if __name__=="__main__":

    ENABLE_FAKE_FLOW_TEST=False # enable fake vision flow -> load images from local rather than camera
    print("!!FAKE FLOW TEST") if ENABLE_FAKE_FLOW_TEST else None
    IMG_COLLECT_ABLE = False
    LOG_IN_EXP = True
    task_select="insert" #screw

    ARM_SELECT_DICT={"insert":"right","screw":"left"}

    arm_select=ARM_SELECT_DICT[task_select]
    # rl_model_name="PPO2_yw_screws__10e4.zip"
        #last insert rl_model_name = PPO2_yw_inss_v10_a4_nobias_rew2_stic_10e4

    # Model Paras
    RL_MODEL_DICT = {"screw": "PPO2_yw_screws_id001_8e4", "insert": "PPO2_yw_inss_v10_a4_nobias_rew2_stic_10e4.zip"}

    STICKY_DICT={"insert":2,"screws":None}
    sticky = STICKY_DICT[task_select]
    SUCCESS_STEPS_DICT={"screw":int(60/sticky),"insert":int(20/sticky)}

    # Motion Paras
    ACTION_SCALE_DONW={"screw":None,"insert":0.1} #scale*0~1 cm per timestamp
    Z_VEL={"screw":None,"insert":0.002}
    MOVE_TIMESTEMP_SECOND={"screw":None,"insert":0.5}

    # Image Paras
    MAX_FORCE_DICT={"screw":1,"insert":13}
    IMG_SIZE_DICT={ "screw":85,"insert":65}

    # Others
    CAM_IMG_SIZE = IMG_SIZE_DICT[task_select] #65 for insert, 90 for screws
    stps = 0
    loop_enable = False
    successs_teps = SUCCESS_STEPS_DICT[task_select]
    rl_model_name=RL_MODEL_DICT[task_select]


    if not ENABLE_FAKE_FLOW_TEST:
        import rospy
        rospy.init_node('RI Control', anonymous=True)
        print(sys.path)
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge, CvBridgeError
        import rospy

        print("import arm controller!")
        from kinova_ct_v2 import arm_control
        print("To Reset...")
        arm = arm_control()

        arm.set_gripper(0)
        print("To Reset...")
        arm.reset(mode=task_select,hard=True)
        print("Reset Fisnihed")

    ri=RealInsert(rl_model_name=rl_model_name,taskselect=task_select)



    while(True):
        action = ri.model.get_action()
        # action=np.mean(np.array([ri.get_action() for i in range(1)]),0)
        print("Get Action=",action)
        if(loop_enable):
            cmd="y"
        else:
            cmd = input("Go On? y-> move, r->reset, c-> control, f-> get force, q-> quite, e-> enable auto loop")

        if cmd=="y":
            a1=action[0]*ACTION_SCALE_DONW[task_select]
            a2=action[1]*ACTION_SCALE_DONW[task_select]
            arm.move_relate(task_select, [a1,a2],maxforce=20,sticky=sticky,z_vel=Z_VEL[task_select],timestamp=MOVE_TIMESTEMP_SECOND[task_select])
            force=arm.get_force_z(arm_select)
            stps+=1
            print("steps=",stps)
            print("force",force)
            if(force>MAX_FORCE_DICT[task_select]):
                print("collision Detect!")
                arm.reset(mode=task_select)
                stps=0
                loop_enable=True

            if(stps>int(successs_teps)):
                print("Max Stps!")
                if task_select=="insert":
                    stps=0
                    # arm.set_gripper(1)
                    # arm.reset(mode=task_select)
                    # arm.set_gripper(0)
                    loop_enable=True

        elif cmd == "e":
            loop_enable=True
        elif cmd=="r":
            arm.reset()
        elif cmd=="c":
            updn=float(input("updn"))
            leri=float(input("leri"))
            arm.move(arm_select,up_dn=updn,le_ri=leri)
        elif cmd =="f":
            force=arm.get_force_z(arm_select)
            print("force =",force)
        elif cmd =="g":
            ri.switch_gan()
        elif cmd=="q":
            break
