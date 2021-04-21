import time
import zmq
import numpy as np
import threading
import os
import sys
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
from stable_baselines3 import PPO
from rl_environment.My_Env import yw_robotics_env

class RLINSERT_Server(object):
    def __init__(self,model_name):
        self.context = zmq.Context()
        self.skt = self.context.socket(zmq.REP)
        self.skt.bind("tcp://*:5700")
        self.model_name=model_name
        self._init_server()

    def _init_server(self):
        task_name = "yw_insf_v1"
        env = yw_robotics_env(task_name, DIRECT=1)
        self.model = PPO.load(self.model_name, env, verbose=1)

    def start(self):
        self.thr1=threading.Thread(target=self.run)
        self.thr1.start()

    def run(self):
        print("RL Insertion Server is running...")
        while True:
            message = self.skt.recv()
            img=np.frombuffer(message,dtype=np.uint8)
            if img.shape==(98304,):
                img=np.reshape(img,[3,128,256])
            if img.shape == (3,128,256):
                pass
            elif img.shape == (128,256,3):
                img=np.transpose(img,[2,0,1])
            else:
                continue

            action=self.rl_forward(img)
            action=action.squeeze().tobytes()
            self.skt.send(action)
    def rl_forward(self,img):
        action, _states = self.model.predict(img)
        return action
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Insert Server')
    parser.add_argument("-n", "--name", required=True)
    args = parser.parse_args()

    rs=RLINSERT_Server(args.name)
    rs.start()