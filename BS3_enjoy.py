
import os
import sys
import time
try:
    rospth1='/opt/ros/kinetic/lib/python2.7/dist-packages'
    rospth2='/home/radoe-1/movo_ws/devel/lib/python2.7/dist-packages'
    # # rospth='/opt/ros/kinetic/lib/python2.7/dist-packages'
    # if rospth in sys.path:
    sys.path.remove(rospth1)
    sys.path.remove(rospth2)
except:
    pass
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
from stable_baselines3 import PPO
from rl_environment.My_Env import yw_robotics_env

vec=False


task_name="yw_insf_v15"
task_name="yw_inss_v10"
task_name="yw_screws"
task_name="alpoderl2"
# model_name="PPO2_insd_mix11111111111111_z_debug1_10.0e4"
# model_name="PPO2_yw_insf_v10_DGX_dualcnn_v3_50.0e4"
# model_name="PPO2_insf_mix33333333330000_DGX_dualcnn_v3_100.0e4"
# model_name="PPO2_yw_insf_v14_DGX_dualcnn_v4_50.0e4"
model_name="PPO2_yw_insf_v15_dgxv62_50.0e4"
model_name="PPO2_yw_srw_v15_screw_v12s_50.0e4"
model_name="PPO2_yw_inss_v10_z_test_pc_100e4"
model_name="PPO2_yw_inss_v10_action4_10e4_"
model_name="PPO2_yw_inss_v10_action4_nobias_10e4"
model_name="PPO2_yw_inss_v10_action4_nobias_rew2_10e4"
# model_name="PPO2_yw_screws_id001_8e4"
model_name="PPO2_yw_screws_id001_8e4"
model_name="PPO2_alpoderl_id001_8e4"
model_name="PPO2_alpoderl2_debug__8e4"
model_name="PPO2_alpoderl2_debug_001_10e4"
model_name="PPO2_alpoderl2_debug_002_middle_200e4"
model_name="PPO2_alpoderl2_debug_002_final_150e4"
# model_name="PPO2_alpoderl2_debug_003_middle_10e4"
# model_name="PPO2_alpoderl2_debug_003_final_10e4"
model_name="PPO2_alpoderl2_debug_004_final_30e4"
model_name="ICCV_v1_alpoderl2_NM_40.0e4"

env = yw_robotics_env(task_name,DIRECT=0, gan_dgx=True,gan_port=5610)

model=PPO.load(model_name, env, verbose=1)
obs = env.reset()

for i in range(10000):
    # print("obs :6",obs[:6])
    # print("obs 6:",obs[6:])
    action, _states = model.predict(obs,deterministic=True)

    # print(i)
    # print(action)
    # action=[1,]
    obs, rewards, dones, info = env.step(action)
    print("obs=", obs)
    # print(rewards)
    env.render()
    # time.sleep(0.1)
    # print("LOOP->",i)