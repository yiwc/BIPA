import time
import zmq
import numpy as np
import os
import multiprocessing
import threading
import sys
import os
import cv2
sys.path.append(os.path.join(os.getcwd(),"yolov4"))
# from yolov4.mymodel import detect_cv2
from yolov4.mymodel import *
class YOLO_SERVER(object):
    def __init__(self,model_name,weight="last"):

        data_root="yolov4/darknet/data/{}".format(model_name)
        weight_root="yolov4/darknet/backup"
        self.cfgfile=os.path.join(data_root,model_name+".cfg")
        self.weightfile=os.path.join(weight_root,"{}_{}.weights".
                                     format(model_name,weight))
        self.namesfile=os.path.join(data_root,model_name+".names")
        self.datafile=os.path.join(data_root,model_name+".data")


        self.m = Darknet(self.cfgfile)
        self.m.print_network()
        self.m.load_weights(self.weightfile)
        print('Loading weights from %s... Done!' % (self.weightfile))

        self.num_classes = self.m.num_classes
        self.class_names = load_class_names(self.namesfile)


        use_cuda=True
        if use_cuda:
            self.m.cuda()

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5600")
        self._init_()
    def start(self):
        self.thr1=threading.Thread(target=self.run)
        self.thr1.start()
    def run(self):
        print("Server is running...")
        skt=self.socket
        while True:
            #  Wait for next request from client
            message = skt.recv()
            img=np.frombuffer(message,dtype=np.uint8)
            img=np.reshape(img,(128,128,3)).astype(np.uint8)

            boxes=self.gen(img)

            skt.send(np.array(boxes).tobytes())

    def _init_(self):

        pass
    def gen(self,img):
        assert img.dtype==np.uint8

        # img = cv2.imread(imgfile)
        sized = cv2.resize(img, (self.m.width, self.m.height))
        # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        for i in range(2):
            start = time.time()
            boxes = do_detect(self.m, sized, 0.4, 0.6, use_cuda)
            finish = time.time()
            if i == 1:
                print('Predicted in %f seconds.' % ((finish - start)))
        print("yolo server boxxes=>",boxes)
        return boxes

if __name__=="__main__":
    import argparse
    args=argparse.ArgumentParser()
    args.add_argument("-n","--name",required=True,type=str)
    g=args.parse_args()
    yolo_server=YOLO_SERVER(g.name,weight="final")
    yolo_server.start()
    pass