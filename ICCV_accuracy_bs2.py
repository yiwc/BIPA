import gym
import os
import sys
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
from rl_environment.My_Env import yw_robotics_env
import numpy as np
import os
import sys
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
from stable_baselines.common.policies import CnnPolicy
# from ppo2 import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
from rl_environment.My_Env import yw_robotics_env
import random
import os
import sys
import time
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
from rl_environment.My_Env import yw_robotics_env
import numpy as np

def show_dict(acc_dict):
    for k1 in acc_dict.keys():
        print(k1)
        score_dict = acc_dict[k1]
        for k2 in score_dict.keys():
            print(k2, score_dict[k2])


file_names=os.listdir()
test_models=[]
dontwants=["fix","Mon"]
wants=["ICCV_v3_alpoderl2","Lstm","nodemo"]
# only_wants=[]
for file_name in file_names:
    # if  in file_name:
    dontwant=0
    want=0
    for d in dontwants:
        if d in file_name:
            dontwant=1

    for w in wants:
        if w in file_name:
            want+=1
    want=want==len(wants)
    if want==1 and dontwant==0:
        test_models.append(file_name)
#
# file_names=os.listdir()
# test_models=[]
# for file_name in file_names:
#     if "ICCV_v3_alpoderl2" in file_name and "Lstm" in file_name:
#         test_models.append(file_name)
# print(test_models)
maxepi=100

nenvs=32
difficulties=["Easy","Final"]
task_name="alpoderl2_nodemo"
acc_dict ={difficulty: {model_name:0 for model_name in test_models} for difficulty in difficulties}
for difficulty in difficulties:
    env = yw_robotics_env(task_name, DIRECT=1, gan_dgx=True, gan_port=5610, difficulty=difficulty)
    for model_name in test_models:
        model=PPO2.load(model_name)
        obs = env.reset()
        success=0

        for epi in range(maxepi):
            print(epi)
            rewards=[]
            reward=0
            state = None
            dones = [False for _ in range(nenvs)]
            while True:
                vec_obs = np.vstack([np.expand_dims(obs, 0) for _ in range(nenvs)])
                action, state = model.predict(vec_obs, state=state, mask=dones)
                action = action.mean(0)
                action = np.squeeze(action)

                # action, _states = model.predict(obs,deterministic=True)
                action = action*(1-reward*reward)
                env.render()
                obs, reward, done, info = env.step(action)
                dones = [done for _ in range(nenvs)]
                rewards.append(reward)

                if done:
                    # print("dones")
                    compare=np.array(rewards)[-5:]
                    # print(compare)
                    this_success=(compare > 0.7).any()
                    success +=this_success
                    if this_success==False:
                        # print("fail")
                        pass
                    acc_dict[difficulty][model_name]=float(success/maxepi)
                    # print("Score=",acc_dict)
                    show_dict(acc_dict)
                    break

        del model
    env.close()
    del env

show_dict(acc_dict)
# for k1 in acc_dict.keys():
#     print(k1)
#     score_dict=acc_dict[k1]
#     for k2 in score_dict.keys():
#         print(k2,score_dict[k2])