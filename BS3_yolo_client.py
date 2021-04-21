import zmq
import numpy as np
class YOLO_CLIENT():
    def __init__(self,dgx=False,pt=5600):
        self.dgx=dgx
        self.pt=pt
        self._init_()
        pass

    def _init_(self):
        ip="localhost" if self.dgx==False else "172.16.127.89"
        self.context = zmq.Context()
        self.skt = self.context.socket(zmq.REQ)
        self.skt.connect("tcp://{}:{}".format(ip,self.pt))

    def gen(self,img):
        assert img.dtype==np.uint8

        self.skt.send(img.tobytes())
        message = self.skt.recv()
        boxes=np.frombuffer(message,dtype=np.float).reshape(-1,7).tolist()
        # print("yolo client get boxes=>",boxes)
        return boxes
if __name__=="__main__":
    a=YOLO_CLIENT()

    # img = np.random.randint(0, 255, [128, 128, 3]).astype(np.uint8)
    # import cv2
    # img = cv2.imread("yolov4/testbolt4.png")
    # img = cv2.resize(img, (128,128))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # print(a.gen(img))
    pass
