import os
import sys
path_env=os.path.join(os.getcwd(),"rl_environment")
sys.path.append(path_env)
import numpy as np
from glob import glob
import pathlib
# from gym.envs.classic_control import rendering
# from gan_client import GAN_CLIENT
import pandas as pd
from BS3_yolo_client import YOLO_CLIENT
import cv2
from stable_baselines3 import PPO


class Enjoyer(object):
    def __init__(self):
        # self.data={}
        # self.data=
        self.g=YOLO_CLIENT()
        # self.viewer=rendering.SimpleImageViewer()
        self.data_list=[pathlib.PurePosixPath(d) for d in glob("./testSet/dualinsert/l*/*",recursive=True)]
        self.data={}

        pd_dict = {"pth": [], "label": [], "eye": [], "img": [], "id": []}

        ids = []
        for d in self.data_list:
            label = d.parts[-2]
            pth = str(d)
            img = cv2.imread(pth, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            name = d.parts[-1]
            eye = "l" if "l" in name else "r"
            id = name.split("_")[0]
            pd_dict["id"].append(id)
            pd_dict["pth"].append(pth)
            pd_dict["label"].append(label)
            pd_dict["eye"].append(eye)
            pd_dict["img"].append(img)
            if id not in ids:
                ids.append(id)
        idimg_dict = {id: [None, None] for id in ids}
        idlabel_dict = {id: None for id in ids}
        for i in range(len(self.data_list)):
            id = pd_dict["id"][i]
            label = pd_dict["label"][i]
            eye = pd_dict["eye"][i]
            img = pd_dict["img"][i]
            lr = 1 if eye == "l" else 0 # revert due to collection error
            idimg_dict[id][lr] = img
            idlabel_dict[id] = label


        self.pop_data=[]
        self.pop_gandata=[]
        self.pop_label=[]
        for id in ids:
            imgs=idimg_dict[id]
            dual_img=[imgs[0],imgs[1]]
            # dual_img = cv2.hconcat([imgs[0], imgs[1]])
            # dual_img = cv2.resize(dual_img, (128, 128))
            # gan_dual_img = cv2.hconcat([self.g.gan_gen(img=imgs[0], ab="ab"), self.g.gan_gen(img=imgs[1], ab="ab")])
            # gan_dual_img = cv2.resize(gan_dual_img, (128, 128))
            label=idlabel_dict[id]
            self.pop_data.append(dual_img)
            # self.pop_gandata.append(gan_dual_img)
            self.pop_label.append(label)

        self.x=self.pop_data
        self.gan_x=self.pop_gandata
        self.y=list(map(lambda x:int(x.strip("l"))-1,self.pop_label))
        self.pred=[]
        pass
    def clear(self):
        del self.pred
        self.pred=[]
    def render(self,img):
        # self.viewer.imshow(img)
        pass
    def action_to_label(self,action):
        y=action[0][0]
        x=action[0][1]
        angle=np.arctan2(y, x) * 180 / np.pi
        if(angle<22.5):
            angle=angle+360
        # print(angle)
        label = (angle-22.5)//45
        return int(label)
    def record_action(self,action):
        label=self.action_to_label(action)
        self.pred.append(label)
    def score(self):
        return accuracy_score(self.y,self.pred)
    def score_fus(self,f):
        true1=[[7,0,1],]
        y=self.y[:]
        p=self.pred[:]
        total=len(y)
        score=0
        for i in range(len(y)):
            yi=y[i]
            pi=p[i]

            candidate=list(range(yi-f,yi+f+1))
            # for i in range(len(candidate)):
            candidate = list(map(lambda x: x + 8 if x < 0 else x, candidate))
            candidate = list(map(lambda x: x - 8 if x > 8 else x, candidate))
            if pi in candidate:
                score+=1
            # if pi in range(yi-f,yi+f+1):
            #     score+=1
            # return score
        return score/total
    def _load_data(self):
        pass

from BS3_yolo_RL_inference_model import inference_rl_model
def test(model_name,rand=False):
    if rand==False:
        # model = PPO2.load(model_name, seed=1)
        # model = PPO.load(model_name)
        model = inference_rl_model(dgx=True,rl_model_name=model_name)
        # model.load_rl(model_name)
    x = enjoyer.x
    for img_pair in x:
        # enjoyer.render(img)
        if rand == False:
        #calculate observes
            for cam,img in enumerate(img_pair):
                boxes=enjoyer.g.gen(img)
                model.input_box_observe(boxes,cam)
            obs=model.get_observe()
            # print("obs :6",obs[:6])
            # print("obs 6:",obs[6:])
        if rand:
            action=(np.random.uniform(-1,1,[2]),None)
        else:
            action = model.predict(obs,deterministic=True)
        print(action[0])
        enjoyer.record_action(action)
    # print(enjoyer.pred)
    # print("pred,true", list(zip(enjoyer.pred, enjoyer.y)))
    # print("Score==", enjoyer.score())
    print(model_name,end=" -->")
    # if(gan):
    #     print("GAN ab: ",end='')
    score1=enjoyer.score_fus(1)
    score2=enjoyer.score_fus(2)
    print("  Score_fus 1==", score1, "  Score_fus 2==", score2)

    enjoyer.clear()
    return model_name,score1,score2

    # vec = False
if __name__=="__main__":
    np.random.seed(seed=1)
    model_names=[
        # "PPO2_yw_inss_v10_z_test_pc_100e4.zip",
        # "PPO2_yw_inss_v10_z_test_pc_2e4.zip",
        "PPO2_yw_inss_v10_action4_10e4_.zip",
    ]

    enjoyer = Enjoyer()
    mydicts=[]
    for n in model_names:
        model_name,score1,score2=test(n)
        mydicts.append({"model_name":model_name,"score1":score1,"score2":score2})
    model_name,score1,score2=test("random",rand=True)
    mydicts.append({"model_name":model_name,"score1":score1,"score2":score2})

    df=pd.DataFrame(mydicts)
    df=df.sort_values(by="score1")
    df.to_csv("BS_enjoy_dual_realdata_rank1.csv")
    df=pd.DataFrame(mydicts)
    df=df.sort_values(by="score2")
    df.to_csv("BS_enjoy_dual_realdata_rank2.csv")


    # enjoyer=Enjoyer()
    #
    # vec=False
    # # env = gym.make('CartPole-v1')
    # # task_name="yw_insert_v1img3cm"
    # # task_name="yw_insert_v2img3cm"
    # # task_name="yw_insert_v3img3cm"
    # model_name="PPO2_yw_insert_v3img3cm_12e4"
    # model_name="PPO2_yw_insert_v4img3cm_cnn_60e4"
    # model_name="PPO2_yw_insert_v4img3cm_cnn_150e4"
    # model_name="PPO2_yw_insert_v4img3cm_cnn_arcdgx_10e4"
    # # model_name="PPO2_yw_insert_v4img3cm_cnn_arcdgx_150e4"
    # # model_name="PPO2_yw_insert_v3img3cm_cnnlnlstm_12e4"
    # # model_name="PPO2_yw_insert_v4img3cm_cnnlnlstm_12e4"
    # # model_name="PPO2_yw_insert_v4img3cm_cnnlnlstm_36e4"
    # # vec=True
    #
    # # env = VecFrameStack(env, n_stack=4)
    #
    # # env = yw_robotics_env(task_name,DIRECT=0)
    #
    # model=PPO2.load(model_name,seed=1)
    #
    # for img in enjoyer.x:
    #     action=model.predict(img)
    #     enjoyer.record_action(action)
    # # print(enjoyer.pred)
    # print("pred,true",list(zip(enjoyer.pred,enjoyer.y)))
    # print("Score==",enjoyer.score())
    # print("Score_fus 1==",enjoyer.score_fus(1))
    # print("Score_fus 2==",enjoyer.score_fus(2))
    # # print("Score_fus 2==",enjoyer.score_fus(3))
    #     # pass
    #     # pass
    # # for i in range(10000):
    # #     action, _states = model.predict(obs)
    # #     if(vec):
    # #         action=np.squeeze(action)
    # #     obs, rewards, dones, info = env.step(action)
    # #     if vec:
    # #         obs=np.expand_dims(obs,0)
    # #     env.render()