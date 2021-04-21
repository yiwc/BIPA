import os
import re
import pandas as pd
print(os.getcwd())
pattern = "2_(.*?)_40"
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy import interpolate
import numpy as np
# import pandas aspd
def plt_dfs(dfs,name="untitle",xlabel="Steps",ylabel="Mean Rewards"):
    id=0
    # fig = matplotlib.pyplot.gcf()
    for k in dfs.keys():
        # dfs[k]=df
        df=dfs[k]
        if id==0:
            ax=df.plot()
        else:
            df.plot(ax=ax)
        # plt.legend("tes{}".format(1))
        id+=1

    plt.grid()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    # plt.xlim()
    # plt.ylim(-10,20)
    plt.legend(dfs.keys())
    plt.title(name)
    plt.savefig(name+".jpg")
    plt.show()

def dfs_to_hisdfs(dfs):
    targetdfs = {}

    def test(row):
        row["num"] = (df["Value"] > row["Value"]).sum()
        return row

    for k in dfs.keys():
        df = dfs[k]
        df_his = pd.DataFrame({"Value": [i for i in range(-10, 18)], "num": [0 for i in range(-10, 18)]})
        # df_his.set_index("Value",inplace=True)
        df_his.apply(test, axis=1)
        df_his.set_index("Value", inplace=True)
        df_his["num"] = df_his["num"] / 100
        print(df_his)
        targetdfs[k] = df_his
    return targetdfs

if __name__=="__main__":
    data_pth=os.path.join(os.getcwd(),"datas")
    files=os.listdir(data_pth)


    fixcam_dfs={}
    randcam_dfs={}
    id=0
    for f in files:
        substring=re.search(pattern,f).group(1)
        df=pd.read_csv(os.path.join(data_pth,f))
        df.drop("Wall time",axis=1,inplace=True)
        df.set_index('Step',inplace=True)
        # df['Value'] = savgol_filter(df['Value'], 27 , 3)
        # df['Value']=np.mean([df["Value"].shift(1),df["Value"].shift(2)])
        # v=
        ori_v=df['Value'].copy()
        av=5
        # for i in range(av):
        #     df['Value']+=ori_v.shift(i).fillna(-5)#,method='backfill')
        # df['Value']=df['Value']/(av+1)
        for i in range(av):
            df["Value"]=(df["Value"]+df["Value"].shift(1).fillna(method='backfill'))/2

        # df['Ori']=ori_v

        if "fix" in substring:
            fixcam_dfs[substring]=df
        else:
            randcam_dfs[substring]=df

    # plt_dfs(fixcam_dfs,name="Camera Fixing Training")
    # plt_dfs(randcam_dfs,name="Camera Randomization Training")
    randcam_histo_dfs=dfs_to_hisdfs(randcam_dfs)
    fixcam_histo_dfs=dfs_to_hisdfs(fixcam_dfs)

    plt_dfs(randcam_histo_dfs,xlabel="Min. Rewards",ylabel="Cumulative Density",name="Ramdomization Camera")
    plt_dfs(fixcam_histo_dfs,xlabel="Min. Rewards",ylabel="Cumulative Density",name="Fixed Camera")