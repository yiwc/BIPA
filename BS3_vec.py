import gym
import os
import sys
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.policies import CnnPolicy
# from ppo2 import PPO2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common import set_global_seeds
import stable_baselines3
# from BS3_policies import CustomCNN
# from BS3_policies import TorchResnet
# from BS3_policies import DualEyeCNN
# from BS3_policies import DualEyeResNet
# from BS3_policies import DualEyeMLP
# from BS3_policies import DualEyeTransformer
# from BS3_policies import DualEyeMLPez
# from BS3_policies import DualEzMap
# from BS3_policies import DualMap
# from BS3_policies import DualMapV2
# from BS3_policies import DualMapV3
from BS3_policies import *
from stable_baselines3.common.callbacks import EvalCallback
# stable_baselines3.ppo.CnnPolicy
from rl_environment.My_Env import yw_robotics_env
import random


class MyRLTrainer(object):
    def __init__(self,args=None):
        if(args is None):
            import argparse
            parser = argparse.ArgumentParser(description='PPO2 with insert task')
            parser.add_argument("-pre","--prefix", default="ICCV_debug")
            parser.add_argument("-t","--task", default="alpoderl2", type=str)  # mix
            parser.add_argument("-e","--num_steps_e4", type=float, default=2)
            parser.add_argument("-c","--cpus", type=int, default=40)
            parser.add_argument("-l","--load_from", default="", type=str)
            parser.add_argument("-p","--policy", default="MonIML", type=str)
            parser.add_argument("--test_env", default="alpoderl2", type=str)
            parser.add_argument("-g","--gan_srvs", default=2, type=int)
            parser.add_argument("-pt","--gan_port", default=2, type=int)
            args = parser.parse_args()
        self.args=args
        self.task_name=args.task
        self.model_name=args.prefix+"_"+self.task_name+"_"+args.policy+"_"+str(args.num_steps_e4)+"e4"
        self.tasks_pool=["yw_insert_v1img3cm","yw_insert_v2img3cm","yw_insert_v3img3cm","yw_insert_v4img3cm","yw_insert_g1img","yw_insert_g1bimg"]+\
                        ["yw_insd_v"+str(i) for i in range(1,16)]+\
                        ["yw_insf_v"+str(i) for i in range(1,20)]+\
                        ["yw_srw_v"+str(i) for i in range(1,16)]+\
                        ["yw_inss_v"+str(i) for i in range(1,16)]+["yw_screws"]+\
                        ['alpoderl','alpoderl2','alpoderl2_fixcam']
        gan_srvs=args.gan_srvs
        self.gan_srvs=gan_srvs
        gan_port=args.gan_port
        self.gan_port=gan_port

        print(self.tasks_pool)
        self.task_weight=[1,2,3,4]

        policy_dict={
            "cnn":CustomCNN,
            "resnet":TorchResnet,
            "dualcnn":DualEyeCNN,
            "dualres":DualEyeResNet,
            "dualmlp":DualEyeMLP,
            "dualmlpez":DualEyeMLPez,
            "DualEyeTransformer":DualEyeTransformer,
            "DualEzMap":DualEzMap,
            "DualMap":DualMap,
            "DualMapV2":DualMapV2,
            "DualMapV3":DualMapV3,
            "IML":IML,
            "NM":NM,
            "PML":PML,
            "MML":MML,
            "MonIML":MonIML,
        }

        self.policy=policy_dict[args.policy]


        self.num_cpu = args.cpus

        if self.task_name=="mix":
            # tasks_pool= ["yw_insert_v1img3cm","yw_insert_v2img3cm","yw_insert_v3img3cm","yw_insert_v4img3cm"]
            import random
            # weights=[2,3,4,5]
            vecs=[self.make_random_env() for i in range(self.num_cpu)]
            vec_env=SubprocVecEnv(vecs)
            self.env = vec_env
        elif self.task_name in self.tasks_pool:
            self.env = SubprocVecEnv([self.make_env(args.task, i) for i in range(self.num_cpu)])
        elif self.task_name[0:8]=="insd_mix" and len(self.task_name)==20:
            mixs=self.task_name[8:20]#8:20
            # mix14=self.task_name[8:21]
            vecs=[]

            env_id=1
            for s in mixs:
                envs=[self.make_x_env("d"+str(env_id)) for i in range(int(s))]
                vecs=vecs+envs
                env_id+=1
            print(vecs)
            vec_env = SubprocVecEnv(vecs)
            self.env = vec_env
        elif self.task_name[0:8]=="insd_mix" or "insf_mix" and len(self.task_name)==22:
            mixs=self.task_name[8:22]#8:20
            # mix14=self.task_name[8:21]
            if "d" in self.task_name:
                flag = "d"
            elif "f" in self.task_name:
                flag="f"
            else:
                flag=None
            vecs=[]
            env_id=1
            for s in mixs[:12]:
                envs=[self.make_x_env(flag+str(env_id)) for i in range(int(s))]
                vecs=vecs+envs
                env_id+=1
            env_id=14
            s=str(mixs[12])
            vecs=vecs+[self.make_x_env(flag+str(env_id)) for i in range(int(s))]
            env_id=15
            s=str(mixs[13])
            vecs=vecs+[self.make_x_env(flag+str(env_id)) for i in range(int(s))]
            print(vecs)
            vec_env = SubprocVecEnv(vecs)
            self.env = vec_env

        else:
            envs_v1=[self.make_x_env("v1") for i in range(int(self.task_name[0]))]
            envs_v2=[self.make_x_env("v2") for i in range(int(self.task_name[1]))]
            envs_v3=[self.make_x_env("v3") for i in range(int(self.task_name[2]))]
            envs_v4=[self.make_x_env("v4") for i in range(int(self.task_name[3]))]
            envs_g1=[self.make_x_env("g1") for i in range(int(self.task_name[4]))]
            envs_g1b=[self.make_x_env("g1b") for i in range(int(self.task_name[5]))]
            envs_g1c=[self.make_x_env("g1c") for i in range(int(self.task_name[6]))]
            vecs=envs_v1+envs_v2+envs_v3+envs_v4+envs_g1+envs_g1b+envs_g1c
            print("vecs=",vecs)
            vec_env=SubprocVecEnv(vecs)
            self.env=vec_env
        os.makedirs("./tf_logs/"+self.model_name,exist_ok=True)

        log_dir="./z_monitor_logs/"+self.model_name
        os.makedirs(log_dir,exist_ok=True)

        # self.env.reward_range=(-50,50)
        # self.env = Monitor(self.env, log_dir)

        if (args.load_from == ""):
            policy_kwargs = dict(
                features_extractor_class=self.policy,
                features_extractor_kwargs=dict(features_dim=128),
            )
            # self.model = PPO(self.policy, self.env, verbose=1, tensorboard_log="./tf_logs/" +self.model_name,nminibatches=16)
            self.model = PPO("CnnPolicy",
                             self.env,
                             verbose=1,
                             tensorboard_log="./tf_logs/" +self.model_name,
                             policy_kwargs=policy_kwargs,
                             n_steps=100)

        else:
            print("Load from<--",args.load_from)
            self.model = PPO.load(args.load_from,
                                  self.env,
                                  verbose=1,
                                  tensorboard_log="./tf_logs/" + self.model_name)

        eval_env=yw_robotics_env(args.test_env,gan_port=gan_port)
        self.eval_callback = EvalCallback(eval_env,
                                          best_model_save_path='./tf_logs/'+self.model_name,
                                     log_path='./tf_logs/'+self.model_name,
                                          eval_freq=25,
                                     deterministic=True,
                                          render=False)


    def make_env(self,task_name, rank, seed=0):
        gan_srvs=self.gan_srvs
        gan_port=self.gan_port
        def _init():
            env = yw_robotics_env(task_name,gan_srvs=gan_srvs,gan_port=gan_port)
        # env.seed(seed + rank)
            return env
        # set_global_seeds(seed)
        return _init
        # return _init(task_name)
    def make_random_env(self,seed=0):
        # task_name="d"
        def _init():
            tasks_pool = ["yw_insert_v1img3cm", "yw_insert_v2img3cm", "yw_insert_v3img3cm", "yw_insert_v4img3cm", "yw_insert_g1img"]
            task=random.choices(tasks_pool,weights=[1,1,2,3,3],k=1)[0]
            env = yw_robotics_env(task)
            return env
        # set_global_seeds(seed)
        return _init
    def make_x_env(self,x):
        gan_srvs=self.gan_srvs
        gan_port=self.gan_port
        def _init_v1():
            env = yw_robotics_env("yw_insert_v1img3cm")
            return env
        def _init_v2():
            env = yw_robotics_env("yw_insert_v2img3cm")
            return env
        def _init_v3():
            env = yw_robotics_env("yw_insert_v3img3cm")
            return env
        def _init_v4():
            env = yw_robotics_env("yw_insert_v4img3cm")
            return env
        def _init_g1():
            env = yw_robotics_env("yw_insert_g1img")
            return env
        def _init_g1b():
            env = yw_robotics_env("yw_insert_g1bimg")
            return env
        def _init_g1c():
            env = yw_robotics_env("yw_insert_g1cimg")
            return env

        def _init_d1():
            env = yw_robotics_env("yw_insd_v1")
            return env
        def _init_d2():
            env = yw_robotics_env("yw_insd_v2")
            return env
        def _init_d3():
            env = yw_robotics_env("yw_insd_v3")
            return env
        def _init_d4():
            env = yw_robotics_env("yw_insd_v4")
            return env
        def _init_d5():
            env = yw_robotics_env("yw_insd_v5")
            return env
        def _init_d6():
            env = yw_robotics_env("yw_insd_v6")
            return env
        def _init_d7():
            env = yw_robotics_env("yw_insd_v7")
            return env
        def _init_d8():
            env = yw_robotics_env("yw_insd_v8")
            return env
        def _init_d9():
            env = yw_robotics_env("yw_insd_v9")
            return env
        def _init_d10():
            env = yw_robotics_env("yw_insd_v10")
            return env
        def _init_d11():
            env = yw_robotics_env("yw_insd_v11",gan_srvs=2,gan_port=gan_port)
            return env
        def _init_d12():
            env = yw_robotics_env("yw_insd_v12",gan_srvs=2,gan_port=gan_port)
            return env
        def _init_d14():
            env = yw_robotics_env("yw_insd_v14",gan_srvs=2,gan_port=gan_port)
            return env
        def _init_d15():
            env = yw_robotics_env("yw_insd_v15",gan_srvs=2,gan_port=gan_port)
            return env

        def _init_f1():
            env = yw_robotics_env("yw_insf_v1")
            return env
        def _init_f2():
            env = yw_robotics_env("yw_insf_v2")
            return env
        def _init_f3():
            env = yw_robotics_env("yw_insf_v3")
            return env
        def _init_f4():
            env = yw_robotics_env("yw_insf_v4")
            return env
        def _init_f5():
            env = yw_robotics_env("yw_insf_v5")
            return env
        def _init_f6():
            env = yw_robotics_env("yw_insf_v6")
            return env
        def _init_f7():
            env = yw_robotics_env("yw_insf_v7")
            return env
        def _init_f8():
            env = yw_robotics_env("yw_insf_v8")
            return env
        def _init_f9():
            env = yw_robotics_env("yw_insf_v9")
            return env
        def _init_f10():
            env = yw_robotics_env("yw_insf_v10")
            return env
        def _init_f11():
            env = yw_robotics_env("yw_insf_v11",gan_srvs=gan_srvs,gan_port=gan_port)
            print("gan_srvs==",gan_srvs)
            return env
        def _init_f12():
            env = yw_robotics_env("yw_insf_v12",gan_srvs=gan_srvs,gan_port=gan_port)
            print("gan_srvs==",gan_srvs)
            return env
        def _init_f14():
            env = yw_robotics_env("yw_insf_v14",gan_srvs=gan_srvs,gan_port=gan_port)
            print("gan_srvs==",gan_srvs)
            return env
        def _init_f15():
            env = yw_robotics_env("yw_insf_v15",gan_srvs=gan_srvs,gan_port=gan_port)
            print("gan_srvs==",gan_srvs)
            return env
        def _init_f16():
            env = yw_robotics_env("yw_insf_v16")
            return env

        def _init_s1():
            env = yw_robotics_env("yw_inss_v1")
            return env
        def _init_s2():
            env = yw_robotics_env("yw_inss_v2")
            return env
        def _init_s3():
            env = yw_robotics_env("yw_inss_v3")
            return env
        def _init_s4():
            env = yw_robotics_env("yw_inss_v4")
            return env
        def _init_s5():
            env = yw_robotics_env("yw_inss_v5")
            return env
        def _init_s6():
            env = yw_robotics_env("yw_inss_v6")
            return env
        def _init_s7():
            env = yw_robotics_env("yw_inss_v7")
            return env
        def _init_s8():
            env = yw_robotics_env("yw_inss_v8")
            return env
        def _init_s9():
            env = yw_robotics_env("yw_inss_v9")
            return env
        def _init_s10():
            env = yw_robotics_env("yw_inss_v10")
            return env
        def _init_yw_screws():
            env = yw_robotics_env("yw_screws")
            return env


        # set_globfz_seeds(seed=0)
        env_dict={
            "v1":_init_v1,
            "v2":_init_v2,
            "v3":_init_v3,
            "v4":_init_v4,
            "g1":_init_g1,
            "g1b":_init_g1b,
            "g1c":_init_g1c,
            "d1":_init_d1,
            "d2":_init_d2,
            "d3":_init_d3,
            "d4":_init_d4,
            "d5":_init_d5,
            "d6":_init_d6,
            "d7":_init_d7,
            "d8":_init_d8,
            "d9":_init_d9,
            "d10":_init_d10,
            "d11":_init_d11,
            "d12":_init_d12,
            "d14":_init_d14,
            "d15":_init_d15,
            "f1":_init_f1,
            "f2":_init_f2,
            "f3":_init_f3,
            "f4":_init_f4,
            "f5":_init_f5,
            "f6":_init_f6,
            "f7":_init_f7,
            "f8":_init_f8,
            "f9":_init_f9,
            "f10":_init_f10,
            "f11":_init_f11,
            "f12":_init_f12,
            "f14":_init_f14,
            "f15":_init_f15,
            "f16":_init_f16,
            # yw_inss
            "s1":_init_s1,
            "s2":_init_s2,
            "s3":_init_s3,
            "s4":_init_s4,
            "s5":_init_s5,
            "s6":_init_s6,
            "s7":_init_s7,
            "s8":_init_s8,
            "s9":_init_s9,
            "s10":_init_s10,
            # yw_screws
            "yw_screws":_init_yw_screws,

        }
        return env_dict[x]

    def default_train(self):
        # eval_env=yw_robotics_env("yw_insd_v15",gan_srvs=1)

        self.model.learn(total_timesteps=int(float(self.args.num_steps_e4)*10000),
                         callback=self.eval_callback,)
        self.model.save(self.model_name)
        print("Model Trained Finished & Saved --> ", self.model_name)

if __name__=="__main__":
    obj=MyRLTrainer()
    obj.default_train()
