# v3
# input is dual image
# go through rl server

BS="BS3"#"BS3" "BS2"

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

from BS3_yolo_RL_inference_model_v3 import inference_rl_model

from threading import Thread

class RealInsert(object):

    def __init__(self,rl_model_name,taskselect,yolo_class_names,predata):
        self.taskselect=taskselect

        self.model=inference_rl_model(dgx=True,
                                      rl_model_name=rl_model_name,
                                      taskselect=taskselect,
                                      yolo_class_names=yolo_class_names,
                                      predata=predata)

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
            self.bridge = CvBridge()
        else:
            self.load_testimgs()

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
            imgs_dir=os.path.join(os.getcwd(),"saveimgs","fakeimgflow")
            imgs_name=os.listdir(imgs_dir)
            # fix_id=-48
            # candidate_names=imgs_name[fix_id:fix_id+2]

            # fix_id=-50
            # candidate_names=imgs_name[fix_id:fix_id+2]
            fix_id=5
            candidate_names=imgs_name[fix_id:fix_id+200]
            while True:
                if fix_id:
                    img_name=random.choice(candidate_names)
                # for img_name in imgs_name:
                    # print(imgs_name)
                    img_f=os.path.join(imgs_dir,img_name)
                    img = cv2.imread(img_f)
                    img = np.array(img).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if "l" in img_name:
                        self.imgs[0]=img
                    elif "r" in img_name:
                        self.imgs[1]=img
                # print("load imgs..")
                    time.sleep(0.3)
        thr = Thread(target=load_imgs, args=(self,))
        thr.start()

    def keep_cap_dualimg(self,seris_id):

        def img_process(img):
            img = cv2.resize(img, (320, 240))
            c_ = [int(img.shape[0] / 2), int(img.shape[1] / 2), 3]
            size = self.cam_img_size # 65 insert
            if RANDOM_FOV:
                size+=random.randint(-RANDOM_FOV,RANDOM_FOV)
                c_[0]+=random.randint(-RANDOM_SHIFT,RANDOM_SHIFT)
                c_[1]+=random.randint(-RANDOM_SHIFT,RANDOM_SHIFT)

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
            self.imgs_captured[0]=frame1
            self.imgs_captured[1]=frame2
            drawed = self.imgs_captured.copy()


            # update sensors based on the imgs
            # update detector
            # if ENABLE_FAKE_FLOW_TEST:
            # o model =>self.model.input_imgs ")
            self.model.input_imgs(self.imgs_captured.copy())

            # update line detector
            # update 3rd points estimation
            dual_img = cv2.hconcat([frame1, frame2])
            key = cv2.waitKey(2)
            if key == 27:  # exit on ESC
                break

            # draw detected img
            boxes_dual=self.model.boxes_dualcamera

            if HUMAN_TEST:
                pass
            else:
                drawed[0]=self.model.tool_draw_box(drawed[0],boxes_dual[0])
                drawed[1]=self.model.tool_draw_box(drawed[1],boxes_dual[1])
                drawed[0]=self.model.tool_draw_line(drawed[0],self.model.lines_detect[0])
                drawed[1]=self.model.tool_draw_line(drawed[1],self.model.lines_detect[1])

                drawed[0]=self.model.tool_draw_point(drawed[0],self.model.hole_pos[0])
                drawed[0]=self.model.tool_draw_point(drawed[0],self.model.refp_pos[0],color=[255,255,0])
                drawed[0]=self.model.tool_draw_point(drawed[0],self.model.head_pos[0])
                drawed[1]=self.model.tool_draw_point(drawed[1],self.model.hole_pos[1])
                drawed[1]=self.model.tool_draw_point(drawed[1],self.model.refp_pos[1],color=[255,255,0])
                drawed[1]=self.model.tool_draw_point(drawed[1],self.model.head_pos[1])
            # drawed[0]=plot_boxes_cv2(drawed[0], boxes_dual[0], class_names=class_names)
            # drawed[1]=plot_boxes_cv2(drawed[1], boxes_dual[1], class_names=class_names)
            #
            # try:
            #     def s(pos):
            #         return  int((pos+1)/2*128)
            #
            #     # hole1x,hole1y= s(self.model.hole_pos[0][0]),s(self.model.hole_pos[0][1])
            #     hole2x,hole2y= s(self.model.hole_pos[1][0]),s(self.model.hole_pos[1][1])
            #     head1x,head1y= s(self.model.head_pos[0][0]),s(self.model.head_pos[0][1])
            #     head2x,head2y= s(self.model.head_pos[1][0]),s(self.model.head_pos[1][1])
            #     # drawed[0] = cv2.rectangle(drawed[0], (hole1x, hole1y), (hole1x, hole1y), [0,0,0], 5)
            #     drawed[0] = cv2.rectangle(drawed[0], (head1x, head1y), (head1x, head1y), [125,0,0], 5)
            #     drawed[1] = cv2.rectangle(drawed[1], (hole2x, hole2y), (hole2x, hole2y), [0,0,0], 5)
            #     drawed[1] = cv2.rectangle(drawed[1], (head2x, head2y), (head2x, head2y), [125,0,0], 5)
            # except Exception as err:
            #     print("tring to mark observation point,",err)
            #     pass

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
    def tool_get_postive_point_pose(self,taskselect):
        if taskselect=="screw":
            return self.model.head_pos[0],self.model.head_pos[1]
        if task_select=="insert":
            return self.model.hole_pos[0],self.model.hole_pos[1]
    def do_collect_demo_vectors(self,arm,sticky):
        ri=self
        task_select=self.taskselect
        input("!Press any key to start collect demo actions : Any key")
        arm.reset(mode=task_select)
        # center_red_cam0=ri.model.head_pos[0]
        # center_red_cam1=ri.model.head_pos[1]
        center_red_cam0,center_red_cam1 = ri.tool_get_postive_point_pose(taskselect=task_select)
            # vl1 vr1
        arm.move_relate(task_select, [1, 0], maxforce=20, sticky=sticky, z_allow=False)
        time.sleep(6)
        print("collect vl1 vr1 ...")

        center_red_cam0_new,center_red_cam1_new = ri.tool_get_postive_point_pose(taskselect=task_select)
        # center_red_cam0_new=ri.model.head_pos[0]
        # center_red_cam1_new=ri.model.head_pos[1]
        vl1 = np.array(center_red_cam0_new) - np.array(center_red_cam0)
        vr1 = np.array(center_red_cam1_new) - np.array(center_red_cam1)
            # vl2 vr2
        arm.reset(mode=task_select)
        arm.move_relate(task_select, [0, 1], maxforce=20, sticky=sticky, z_allow=False)
        time.sleep(6)
        print("collect vl2 vr2 ...")
        # center_red_cam0_new=ri.model.head_pos[0]
        # center_red_cam1_new=ri.model.head_pos[1]
        center_red_cam0_new,center_red_cam1_new = ri.tool_get_postive_point_pose(taskselect=task_select)
        vl2 = np.array(center_red_cam0_new) - np.array(center_red_cam0)
        vr2 = np.array(center_red_cam1_new) - np.array(center_red_cam1)
        print("vl1 vl2 vr1 vr2",vl1,vl2,vr1,vr2,"\n Please log the above vector into pre_observe data for next time use")
        ri.model.input_demo_vector(vl1=vl1,vl2=vl2,vr1=vr1,vr2=vr2)
        print(" finished")
        input("!Press any key to stop collect demo actions : Any key")

    def change_task(self,target_task):
        self.model.change_task(target_task)

        self.sticky = STICKY_DICT[target_task]
        self.arm_select = ARC_SELECT_DICT[target_task]
        self.cam_img_size = IMG_SIZE_DICT[target_task]  # 65 for insert, 90 for screws
        # self.successs_teps = SUCCESS_STEPS_DICT[target_task]
        # self.rl_model_name = RL_MODEL_DICT[target_task]
        arm.reset(target_task,hard=True)
    def do_task(self, target_task, align_err_thresh=0.07,success_quit=True):
        assert target_task in ["screw","insert"]

        self.change_task(target_task)
        time.sleep(2)
        stps = 0
        align_err = 1


        while (True):
            action = self.model.get_action()
            print("Get Action={}, Target Task={}, Model Task={}".format(action,target_task,self.taskselect) )
            a1 = action[0] / 4 * align_err
            a2 = action[1] / 4 * align_err
            arm.move_relate(target_task, [a1, a2], maxforce=20, sticky=self.sticky, z_allow=False)
            force = arm.get_force_z(self.arm_select)
            stps += 1
            print("steps={},Force={}".format(stps,force))
            # print("force", force)
            time.sleep(1)

            align_err=self.get_align_err()
            print("Allignment Error=align_err {}".format(align_err))
            if align_err < align_err_thresh:
                if target_task == "screw":
                    arm.move_down(self.arm_select, d=0.035, f=25)
                    if success_quit:
                        return True
                    arm.reset(mode=target_task)
                    # else:
                if target_task == "insert":
                    arm.move_down(self.arm_select, d=0.05, f=25)
                    arm.set_gripper(self.arm_select,1)
                    # time.sleep(2)
                    arm.move_up(self.arm_select, d=0.05, f=25)
                    arm.set_gripper(self.arm_select,0)
                    if success_quit:
                        return True
                    arm.reset(mode=target_task)

                stps = 0


            # elif cmd == "e":
    def get_align_err(self):
        observations = self.model.observe.copy()
        l2 = observations[3:5, :]
        r2 = observations[7:9, :]
        lt = l2[0:1, :] - l2[1:2, :]
        rt = r2[0:1, :] - r2[1:2, :]
        align_err = np.max([np.linalg.norm(lt), np.linalg.norm(rt)])
        return align_err

if __name__=="__main__":

    # Static Configurations
    RL_MODEL_DICT = {"screw": "PPO2_alpoderl2_debug_004_final_30e4",
                     "insert": "PPO2_alpoderl2_debug_004_final_30e4"}
    ARC_SELECT_DICT = {"screw": "left","insert": "right"}
    PRE_OBSERVE = {
        ## Higher Pitch pos
        "screw": {"line": [np.array([[60, 0.57]]), np.array([[60, -0.53]])],
                  "map": [np.array([0.438, 0.099]), np.array([0.088, 0.622]),
                          np.array([0.650, 0.180]), np.array([-0.008, 0.627])]},
        #           "map": [np.array([0.381, 0.291]), np.array([-0.132, 0.813]),
        #                   np.array([0.837, 0.299]), np.array([0.056, 0.789])]},
        # # l1 l2 r1 r2
        # [0.38178688 0.29130736][-0.1323005   0.81310631] [0.83727568 0.29946014][0.05609055 0.78999218]
        #             "map": [np.array([0.418, 0.099]), np.array([-0.045, 0.654]),
        #                     np.array([0.624, 0.087]), np.array([0.079, 0.108])]},  # l1 l2 r1 r2

    # [0.43858445 0.09995836][0.08849794
    # 0.62215227] [0.65009278 0.18039569][-0.00811344
    # 0.62722641]
        "insert": {"line": [np.array([[60, -0.65]]), np.array([[60, +0.67]])],
                   "map": [np.array([-0.029, 0.893]), np.array([-0.628, 0.137]),
                           np.array([-0.072, 0.942]), np.array([-0.917, 0.159])]},  # l1 l2 r1 r2
        # [-0.02914584  0.89325279][-0.6282897  0.1374414] [-0.07212448  0.94168049][-0.91733956 0.15931737]
        # "insert":{"line":[],"map_left":[],"map_right":[]}
    }
    STICKY_DICT = {"screw": 3, "insert": 3}
    SUCCESS_STEPS_DICT = {"screw": 200, "insert": 200}
    MAX_FORCE_DICT = {"screw": 1, "insert": 13}
    IMG_SIZE_DICT = {"screw": 80, "insert": 75}
    YOLO_CLASS_NAME_DICT = {
        "screw": ["hole", "bolt_head", "bolt_bottom", "screw_head"],
        "insert": ["hole", "bolt_head", "bolt_bottom", "screw_head"]
    }

    # Dynamic Configurations
    task_select="screw" #screw insert
    # Debug Configurations
    HUMAN_TEST=False # allow human input
    PLAY_SEQUENCE=False # (debugging) Run Insert and Screw in sequence
    ENABLE_FAKE_FLOW_TEST=False # enable fake vision flow -> load images from local rather than camera
    print("!!FAKE FLOW TEST") if ENABLE_FAKE_FLOW_TEST else None
    IMG_COLLECT_ABLE = False if ENABLE_FAKE_FLOW_TEST == False else False
    LOG_IN_EXP = True
    COLLECT_DEMO_VECTOR=False
    TRAIN_DATA_COLLECTION=False # collect trainable data images
    RANDOM_FOV = 0
    RANDOM_SHIFT = 0
    if IMG_COLLECT_ABLE and TRAIN_DATA_COLLECTION:
        RANDOM_FOV = 30
        RANDOM_SHIFT = 10

    # Update Configuration from Static Configs
    sticky = STICKY_DICT[task_select]
    arm_select=ARC_SELECT_DICT[task_select]
    CAM_IMG_SIZE = IMG_SIZE_DICT[task_select] #65 for insert, 90 for screws
    successs_teps = SUCCESS_STEPS_DICT[task_select]
    rl_model_name=RL_MODEL_DICT[task_select]


    # init a loop
    stps = 0
    loop_enable = False
    align_err = 1

    if not ENABLE_FAKE_FLOW_TEST:
        import rospy
        rospy.init_node('RI Control', anonymous=True)
        print(sys.path)
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge, CvBridgeError
        print("import arm controller!")
        from kinova_ct_v2 import arm_control
        print("To Reset...")
        arm = arm_control()
        arm.set_grippers(0)
        print("To Reset...")
        arm.reset(mode=task_select,hard=True)
        print("Reset Fisnihed")

    ri=RealInsert(rl_model_name=rl_model_name,
                  taskselect=task_select,
                  yolo_class_names=YOLO_CLASS_NAME_DICT[task_select],
                  predata=PRE_OBSERVE[task_select])

    # collect demo vectors
    if not COLLECT_DEMO_VECTOR or ENABLE_FAKE_FLOW_TEST:

        map=PRE_OBSERVE[task_select]["map"]
        ri.model.input_demo_vector(vl1=map[0], vl2=map[1], vr1=map[2], vr2=map[3])
    else:
        ri.do_collect_demo_vectors(arm=arm,sticky=sticky)


    # ri.do_task("insert")
    # ri.do_task("screw")

    while(True):
        action = ri.model.get_action()
        if HUMAN_TEST:

            while (True):
                try:
                    action=[float(input("a1 (-1,1) = ")),float(input("a2 = "))]
                    break
                except:
                    pass

        print("Get Action=",action)
        if(loop_enable or HUMAN_TEST):
            cmd="y"
        else:
            cmd = input("Go On? y-> move, r->reset, c-> control, f-> get force, q-> quite, e-> enable auto loop")

        if cmd=="y":
            # align_err=1
            # a1=action[0]/4+0.05
            # a2=action[1]/4
            a1=action[0]/4 * align_err
            a2=action[1]/4 * align_err
            # a1=0
            # a2=0.5
            if ENABLE_FAKE_FLOW_TEST:
                continue
            arm.move_relate(task_select, [a1,a2],maxforce=20,sticky=sticky,z_allow=False)
            force=arm.get_force_z(arm_select)
            stps+=1
            print("steps=",stps)
            print("force",force)
            # if(force>MAX_FORCE_DICT[task_select]):
            #     print("collision Detect!")
            #     arm.reset(mode=task_select)
            #     stps=0
            #     loop_enable=True

            # if screw allignment success

            time.sleep(1)

            Human_success=False
            if HUMAN_TEST:
                while(True):
                    try:
                        Human_success=bool(int(input("Success?1/0 = ")))
                        break
                    except:
                        pass
            if task_select in ["screw","insert"]:
                observations=ri.model.observe.copy()
                l2 = observations[3:5, :]
                r2 = observations[7:9, :]
                lt = l2[0:1, :] - l2[1:2, :]
                rt = r2[0:1, :] - r2[1:2, :]
                align_err=np.max([np.linalg.norm(lt),np.linalg.norm(rt)])
                print("Allignment Error={},{},{}".format(lt,rt,align_err))
                if align_err<0.07:
                    success=True
                else:
                    success=False
                if HUMAN_TEST:
                    success=Human_success
                if success:
                    if task_select=="screw":
                        arm.move_down(arm_select,d=0.035,f=25)
                        arm.reset(mode=task_select,noise=True)
                    if task_select == "insert":
                        arm.move_down(arm_select, d=0.05, f=25)
                        time.sleep(2)
                        arm.move_up(arm_select, d=0.05, f=25)
                        arm.reset(mode=task_select)
                    stps = 0
                    loop_enable = True

            # if task_select=="insert":

            # if(stps>int(successs_teps)):
            #     print("Max Stps!")
            #     if task_select=="insert":
            #         stps=0
            #         arm.set_gripper(1)
            #         arm.reset(mode=task_select)
            #         arm.set_gripper(0)
            #         loop_enable=True
            #     elif task_select=="screw":
            #         stps = 0
            #         # arm.screw()
            #         arm.reset(mode=task_select)
            #         loop_enable=True

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
