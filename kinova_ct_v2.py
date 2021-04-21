from movo_msgs.msg import JacoCartesianVelocityCmd
from hrclib.hrclib_client_v6 import odyssey_Interface as odyssey
import rospy
import sys
import time

import movo_arc_lib.msg as arcmsg
import actionlib
import numpy as np

import time
class arm_control(object):
    def __init__(self):

        self.odyssey=odyssey()

        self.arm_pub = [0] * 2
        # self.init_arm_pub =[0]*2

        self.insert_speed = 0.05
        self.cartesianforce_left = [0, 0, 0, 0, 0, 0]
        self.cartesianforce_right = [0, 0, 0, 0, 0, 0]

        # 0: left, 1: right
        self.arm_pub[0] = rospy.Publisher('/movo/left_arm/cartesian_vel_cmd', JacoCartesianVelocityCmd, queue_size=10)
        self.arm_pub[1] = rospy.Publisher('/movo/right_arm/cartesian_vel_cmd', JacoCartesianVelocityCmd, queue_size=10)

        self.default_pose_pickready = [0.642, -0.936, 0.767, -1.46, -0.928, 1.68, -3.14 / 4 - 0.2, \
                                       -0.642, 0.936, -0.767, 1.46, 0.928, -1.68, 3.14 / 4 + 0.2, \
                                       0.46, 0, 0]
        self.default_pose_ready_using_screw_driver = [0.642, -0.936, 0.767, -1.46, -0.928, 1.68, -3.14 / 4 - 0.2,
                                                      -0.477, +0.925, +0.55, +1.550, -1.166, -1.749,
                                                      -0.704 + 3.14 - 0.1,
                                                      0.46, 0, -0.6]
        self.default_jps = {
            "insert": self.default_pose_pickready,
            "screw": self.default_pose_ready_using_screw_driver
        }
        self._upper_body_joints = ["right_shoulder_pan_joint",
                                   "right_shoulder_lift_joint",
                                   "right_arm_half_joint",
                                   "right_elbow_joint",
                                   "right_wrist_spherical_1_joint",
                                   "right_wrist_spherical_2_joint",
                                   "right_wrist_3_joint",
                                   "left_shoulder_pan_joint",
                                   "left_shoulder_lift_joint",
                                   "left_arm_half_joint",
                                   "left_elbow_joint",
                                   "left_wrist_spherical_1_joint",
                                   "left_wrist_spherical_2_joint",
                                   "left_wrist_3_joint",
                                   "linear_joint",
                                   "pan_joint",
                                   "tilt_joint"]




        # # move joint safe
        # name = "L0_upper_jp_move_safe"
        # action = arcmsg.upper_jp_movo_safeAction
        # self.upper_move_client = actionlib.SimpleActionClient(name, action)
        # self.upper_move_client.wait_for_server()

        # # move joint safe
        # name = "L0_dual_task_move_safe_relate"
        # action = arcmsg.dual_task_move_safe_relateAction
        # self.dual_rela_client = actionlib.SimpleActionClient(name, action)
        # self.dual_rela_client.wait_for_server()

        # # move joint safe
        # name = "L0_dual_set_gripper"
        # action = arcmsg.dual_set_gripperAction
        # self.dual_gripper_client = actionlib.SimpleActionClient(name, action)
        # self.dual_gripper_client.wait_for_server()

    def set_gripper(self, rl,value):
        # self.odyssey._L0_dual_set_gripper(value)
        self.odyssey.grip(rl=rl,v=value)
    def set_grippers(self,value):
        self.odyssey._L0_dual_set_gripper(value)
        # self.odyssey.grip(rl=rl,v=value)

    def dual_move_down(self,d,f=30):
        self.odyssey.dual_move_relate(rmove=[0, 0, -d], lmove=[0, 0, -d], time=2, rmaxforce=[f for i in range(6)],
                                      lmaxforce=[f for i in range(6)], wait=True, hard=False)

    def reset(self, mode,hard=True,f=None, noise=False):
        assert mode in ["insert", "screw"]

        duration = 300

        f=f
        hard=hard
        l_max_force = [f for i in range(6)]
        r_max_force = [f for i in range(6)]

        # d=self.odyssey.gval.default_pose_ready_insert_and_screw
        # d=self.odyssey.gval.default_insert1
        # self.odyssey._L0_upper_jp_move_safe(
        #     d["l"],d['r'],d['h'],d['lin'],duration,l_max_force,r_max_force,wait=True,hard=hard
        # )

        if mode == "insert":
            self.odyssey.go_upper_default_jp(self.odyssey.gval.default_insert1,hard=hard,f=f)
            # self.odyssey.dual_move_relate(rmove=[0,0,-0.05],lmove=[0,0.05,0],time=1,rmaxforce=[20 for i in range(6)],lmaxforce=[20 for i in range(6)],wait=True,hard=False)

            self.odyssey.dual_move_relate(rmove=[0,0.06,0],
                                          lmove=[0,0.06,0],
                                          time=1,
                                          rmaxforce=[40 for i in range(6)],
                                          lmaxforce=[40 for i in range(6)],
                                          wait=True,
                                          hard=False)
            self.odyssey.dual_move_relate(rmove=[np.random.uniform(-2,2,1)[0]*0.01,
                                                 np.random.uniform(-2,2,1)[0]*0.01,
                                                 -0.05],
                                          lmove=[0,0.05,0],time=1,rmaxforce=[20 for i in range(6)],lmaxforce=[20 for i in range(6)],wait=True,hard=False)
        if mode == "screw":
            self.odyssey.go_upper_default_jp(self.odyssey.gval.default_pose_screw_rightarm_observe,hard=hard,f=f)
            self.odyssey.go_upper_default_jp(self.odyssey.gval.default_pose_screw_rightarm_observe,hard=hard,f=f)
            time.sleep(1)
            self.odyssey.dual_move_relate(rmove=[0,-0.06,0],
                                          lmove=[0,-0.06,0],
                                          time=1,
                                          rmaxforce=[40 for i in range(6)],
                                          lmaxforce=[40 for i in range(6)],
                                          wait=True,
                                          hard=False)


            if noise:
                move_down=[np.random.uniform(-2.5, 2.5, 1)[0] * 0.01,
                                                     np.random.uniform(-2.5, 2.5, 1)[0] * 0.01,
                                                     -0.02]
                self.odyssey.dual_move_relate(rmove=[0,0,0],
                                              lmove=move_down, time=1, rmaxforce=[20 for i in range(6)],
                                              lmaxforce=[20 for i in range(6)], wait=True, hard=False)
            else:

                self.move_down("left", d=0.02, f=25)

            # self.odyssey.dual_move_relate(rmove=[0,0,0],
            #                               lmove=[0.05,-0.05,0],
            #                               time=1,
            #                               rmaxforce=[20 for i in range(6)],
            #                               lmaxforce=[20 for i in range(6)],wait=True,hard=False)


        return True

    # def dual_move_relate(self,rmove,l):

    def move_relate(self, task_select, action, maxforce=10,sticky=1,z_allow=True,z_vel=0.001,timestamp=1):
        # action = [-1, 1]
        action = list(map(lambda x: x / 100 * sticky, action))
        lmove = [0 for i in range(3)]
        rmove = [0 for i in range(3)]
        if task_select=="screw":
        # if (arm == "left"):
            lmove[0] = action[0]
            lmove[1] = -action[1]
            if z_allow:
                lmove[2] = -z_vel*sticky
            # rmove[0] = action[0]  # x
            # rmove[1] = -action[1]  # y
            # rmove[2] = -0.001 * sticky

        elif task_select=="insert":
        # elif (arm == "right"):
            rmove[0] = action[0] # x
            rmove[1] = -action[1] # y

            if z_allow:
                rmove[2] = -z_vel*sticky
        else:
            raise NotImplementedError("task not defined")
        time = timestamp

        rmaxforce=lmaxforce=[maxforce for i in range(6)]
        self.odyssey._L0_dual_task_move_safe_relate(
            rmove,lmove,time,rmaxforce,lmaxforce)

        # mygoal = arcmsg.dual_task_move_safe_relateGoal()
        # mygoal.pos_r = rmove
        # mygoal.pos_l = lmove
        # mygoal.time = time
        # mygoal.r_max_force = [10 for i in range(6)]
        # mygoal.l_max_force = [10 for i in range(6)]
        #
        # # send a goal
        # self.dual_rela_client.send_goal(mygoal)
        # self.dual_rela_client.wait_for_result()
        print("Move Finished")


    def _subscribe_force_callback_left(self, data):
        # fx = data.x
        # fy = data.y
        # fz = data.z
        # fr = data.theta_x
        # fp = data.theta_y
        # fa = data.theta_z
        # print data
        self.cartesianforce_left[0] = data.x
        self.cartesianforce_left[1] = data.y
        self.cartesianforce_left[2] = data.z
        self.cartesianforce_left[3] = data.theta_x
        self.cartesianforce_left[4] = data.theta_y
        self.cartesianforce_left[5] = data.theta_z
        # self.cartesianforce_left = [fx, fy, fz, fr, fp, fa]

    def _subscribe_force_callback_right(self, data):
        fx = data.x
        fy = data.y
        fz = data.z
        fr = data.theta_x
        fp = data.theta_y
        fa = data.theta_z
        # print data
        self.cartesianforce_right = [fx, fy, fz, fr, fp, fa]

    def get_force_z(self, arm):
        assert arm in ["left", "right"]

        rospy.Subscriber("/movo/left_arm/cartesianforce", JacoCartesianVelocityCmd,
                         self._subscribe_force_callback_left)
        rospy.Subscriber("/movo/right_arm/cartesianforce", JacoCartesianVelocityCmd,
                         self._subscribe_force_callback_right)

        if arm == "left":
            return self.cartesianforce_left[1]
        else:
            return self.cartesianforce_right[1]

    def move_down(self,arm,d,f):
        pos = [0, 0, -d]
        orn = [0, 0, 0, 0]
        maxforce = [f for i in range(6)]
        time=3
        self.odyssey.single_move_relate(arm=arm, move=pos, maxforce=maxforce, time=time, wait=True, hard=False)
    def move_up(self,arm,d,f):
        self.move_down(arm,-d,f)
    def movo_collect_dataset(self,arm):
        time=2
        def move_xy(self,arm,xy):
            pos=[xy[0],xy[1],0]
            # orn=[0,0,0,0]
            maxforce=[20 for i in range(6)]
            self.odyssey.single_move_relate(arm=arm,move=pos,maxforce=maxforce,time=time,wait=True,hard=False)


        def move_down_and_up(self,arm,d):
            pos=[0,0,-d]
            orn=[0,0,0,0]
            maxforce=[20 for i in range(6)]
            self.odyssey.single_move_relate(arm=arm,move=pos,maxforce=maxforce,time=time,wait=True,hard=False)
            pos=[0,0,d]
            self.odyssey.single_move_relate(arm=arm,move=pos,maxforce=maxforce,time=time,wait=True,hard=False)

        def move_down(self,arm,d):
            pos = [0, 0, -d]
            orn = [0, 0, 0, 0]
            maxforce = [20 for i in range(6)]
            self.odyssey.single_move_relate(arm=arm, move=pos, maxforce=maxforce, time=time, wait=True, hard=False)
        def move_up_and_xy(self,arm,d,xy):
            pos = [xy[0], xy[1], d]
            orn = [0, 0, 0, 0]
            maxforce = [20 for i in range(6)]
            self.odyssey.single_move_relate(arm=arm, move=pos, maxforce=maxforce, time=time, wait=True, hard=False)

        # down and up
        d=0.07
        # s=0.0

        xs=[0.01,0.00,-0.01] # front to back to back
        ys=xs.copy() # left to right



        scan_poses=[]
        # for y in ys:
        # scan_poses.append([ys[0], xs[0]])
        scan_poses.append([xs[0], ys[0]])
        scan_poses.append([0, -0.01])
        scan_poses.append([0, -0.01])
        scan_poses.append([0, -0.01])
        scan_poses.append([0, -0.01])
        for i in range(len(xs)-1):
            scan_poses.append([-0.01, +0.04])
            scan_poses.append([0, -0.01])
            scan_poses.append([0, -0.01])
            scan_poses.append([0, -0.01])
            scan_poses.append([0, -0.01])


        for xy in scan_poses:
            d=-d
            # move_xy(self,arm,xy)
            # move_down(self,arm,d)
            # move_up_and_xy(self,arm,d,xy)
            move_up_and_xy(self,arm,d,[0,0]) #move up and down
            # if d<0:
            #     time.sleep(1)
            # move_down_and_up(self,arm,d)

    def __getattr__(self, item):
        return eval("self.odyssey."+item)

if __name__ == '__main__':
    arm = arm_control()
    ARM = "left"

    arm.reset()
    while (True):
        arm.move_relate("right", [0.5, 0])
        arm.move_relate("right", [0.2, 0])

    '''
    step=0
    while True:
        step+=1
        img=get_img_from_cam()
        action = algorithm(img) # action=[0.5,0.5]
        arm.move(ARM,action[0],action[1])

        force_detect = arm.get_force_z(ARM)
        if(force_detect>5):
            arm.reset()
            raw_input("===== Press Enter to Restart =====")
        if(step>100):
            arm.reset()
    '''

    rospy.spin()



