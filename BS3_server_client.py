import zmq
import numpy as np
import time
class RL_CLIENT():
    def __init__(self,dgx=False):
        self.dgx=dgx
        ip="localhost" if self.dgx==False else "172.16.127.89"
        self.context = zmq.Context()
        self.skt = self.context.socket(zmq.REQ)
        self.skt.connect("tcp://{}:5700".format(ip))

    def get_action(self,img):
        img=img.astype(np.uint8)
        self.skt.send(img.tobytes())
        message = self.skt.recv()
        action=np.frombuffer(message,dtype=np.float32)
        return action

if __name__=="__main__":
    cli=RL_CLIENT()
    while True:
        time.sleep(1)
        a=cli.get_action(np.ones([3,128,256],dtype=np.uint8))
        print(a)