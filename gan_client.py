import zmq
import numpy as np
class GAN_CLIENT():
    def __init__(self,dgx=False,pt=5600):
        self.dgx=dgx
        self.pt=pt
        self._init_GAN()
        pass

    def _init_GAN(self):
        ip="localhost" if self.dgx==False else "172.16.127.89"

        self.context = zmq.Context()
        self.sab = self.context.socket(zmq.REQ)
        self.sab.connect("tcp://{}:{}".format(ip,self.pt))
        self.sba = self.context.socket(zmq.REQ)
        self.sba.connect("tcp://{}:{}".format(ip,self.pt+1))

    def gan_gen(self,img,ab):
        # img=np.ones([128,128,3])
        assert ab in ["ab","ba"]
        if ab == "ab":
            skt=self.sab
        else:
            skt=self.sba
        img=img.astype(float)
        skt.send(img.tobytes())
        message = skt.recv()
        genimg=np.frombuffer(message,dtype=np.uint8).reshape([128,128,3])
        return genimg
