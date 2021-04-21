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
# from pybullet_env.My_Env import yw_robotics_env
from rl_environment.My_Env import yw_robotics_env
import random
# from BS_policies import resv2

vec=False
taskname="alpoderl"
# modelname="PPO2_alpoderl_id001_100e4"
modelname="PPO2_alpoderl_id002_200e4"
modelname="PPO2_alpoderl_id001_1000e4"

env = yw_robotics_env(taskname,DIRECT=0)

# if(vec==True):
#     model = PPO2("MlpLstmPolicy", env, verbose=1,nminibatches=1)
#     vec_env = model.get_env()
#     del model
nenvs=32
model=PPO2.load(modelname)
obs = env.reset()
state = None
done = [False for _ in range(nenvs)]

for i in range(10000):
    vec_obs=np.vstack([obs for _ in range(nenvs)])
    action, state = model.predict(vec_obs, state=state, mask=done)
    print(action)
    action=action.mean(0)
    action=np.squeeze(action)
    obs, rewards, done, _ = env.step(action)
    done = [done for _ in range(nenvs)]

    # if vec:
    # obs=np.expand_dims(obs,0)
    env.render()