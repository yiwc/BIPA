import gym
import os
import sys
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
from stable_baselines.common.policies import CnnPolicy
# from ppo2 import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
# from pybullet_env.My_Env import yw_robotics_env
from rl_environment.My_Env import yw_robotics_env
import random
import BS2_policies

class MyRLTrainer(object):
    def __init__(self,args=None):
        if(args is None):
            import argparse
            parser = argparse.ArgumentParser(description='BS Training Scripts')
            parser.add_argument("-pre","--prefix", default="id001")
            parser.add_argument("-t","--task", default="alpoderl2_nodemo", type=str)  # mix
            parser.add_argument("-e","--num_steps_e4", type=float, default=5000)
            parser.add_argument("-c","--cpus", type=int, default=32)
            parser.add_argument("-l","--load_from", default="", type=str)
            parser.add_argument("-p","--policy", default="MlpLstmPolicy", type=str)
            args = parser.parse_args()

        self.args=args
        self.task_name=args.task
        # self.model_name="PPO2_"+self.task_name+"_"+args.prefix+"_"+str(args.num_steps_e4)+"e4"
        self.model_name=args.prefix+"_"+self.task_name+"_"+args.policy+"_"+str(args.num_steps_e4)+"e4"
        #"PPO2_"+self.task_name+"_"+args.prefix+"_"+str(args.num_steps_e4)+"e4"
        self.tasks_pool=["alpoderl2","alpoderl2_fixcam","alpoderl2_nodemo"]
        print(self.tasks_pool)

        POLICY_DICT={
            "MlpLstmPolicy":'MlpLstmPolicy',
            "P1":BS2_policies.P1
        }
        self.policy=POLICY_DICT[args.policy]
        self.num_cpu = args.cpus

        assert self.task_name in self.tasks_pool
        self.env = SubprocVecEnv([self.make_env(args.task, i) for i in range(self.num_cpu)])
        os.makedirs("./tf_logs/"+self.model_name,exist_ok=True)

        if (args.load_from == ""):
            self.model = PPO2(self.policy,
                              self.env,
                              verbose=1,
                              tensorboard_log="./tf_logs/" +self.model_name,
                              nminibatches=16)
        else:
            print("Load from<--",args.load_from)
            self.model = PPO2.load(args.load_from, self.env, verbose=1, tensorboard_log="./tf_logs/" + self.model_name)

    def make_env(self,task_name, seed=0):
        def _init():
            env = yw_robotics_env(task_name)
            return env
        set_global_seeds(seed)
        return _init
    def default_train(self):
        self.model.learn(total_timesteps=int(float(self.args.num_steps_e4)*10000))
        self.model.save(self.model_name)
        print("Model Trained Finished & Saved --> ", self.model_name)

if __name__=="__main__":
    obj=MyRLTrainer()
    obj.default_train()
    print("finsih trained")
    obj.env.close()
    # obj.model.env.close()
    print("close")