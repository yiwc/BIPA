
import os
import sys
import time
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
from stable_baselines3 import PPO
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
dontwants=["Lstm","fix","Mon"]
wants=["ICCV_v3_alpoderl2","IML"]
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
print(test_models)
maxepi=100


difficulties=["Final"]
task_name="alpoderl2"
acc_dict ={difficulty: {model_name:0 for model_name in test_models} for difficulty in difficulties}
for difficulty in difficulties:
    env = yw_robotics_env(task_name, DIRECT=1, gan_dgx=True, gan_port=5610, difficulty=difficulty)
    for model_name in test_models:
        model=PPO.load(model_name, env, verbose=1)
        obs = env.reset()
        success=0

        for epi in range(maxepi):
            print(epi)
        # for i in range(10000):
            rewards=[]
            reward=0
            # print("epi={},sccuess={},model={}".format(epi,success,model_name))
            while True:
                action, _states = model.predict(obs)
                action = action*(1-reward)
                env.render()
                obs, reward, dones, info = env.step(action)

                rewards.append(reward)

                if dones:
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