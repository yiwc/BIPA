import time
import zmq
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import os
import multiprocessing
import threading
class GAN_SERVER(object):
    def __init__(self):
        self.context = zmq.Context()
        self.socket_ab = self.context.socket(zmq.REP)
        self.socket_ba = self.context.socket(zmq.REP)
        self.socket_ab.bind("tcp://*:5600")
        self.socket_ba.bind("tcp://*:5601")
        self._init_GAN()
    def start(self):
        self.thr1=threading.Thread(target=self.run_ab)
        self.thr2=threading.Thread(target=self.run_ba)
        # self.thr1=multiprocessing.Process(target=self.run_ab)
        # self.thr2=multiprocessing.Process(target=self.run_ba)
        self.thr1.start()
        self.thr2.start()
    def run(self,ab):
        print("Gan Server is running...",ab)
        if(ab=="ab"):
            skt=self.socket_ab
        else:
            skt=self.socket_ba
        while True:
            #  Wait for next request from client
            message = skt.recv()
            img=np.frombuffer(message)
            #  Do some 'work'
            img=np.reshape(img,(128,128,3)).astype(np.uint8)
            # time.sleep(1)
            genimg=self.gan_gen(np.expand_dims(img,0),ab)
            genimg=genimg.squeeze().tobytes()
            # print("Transfered!")
            #  Send reply back to client
            skt.send(genimg)
        # pass
    def run_ab(self):
        self.Gab = load_model(os.path.join("GAN_models", self.gab_name), custom_objects=self._objd)
        self.Gab._make_predict_function()
        print("   Gab Model loaded")
        self.run("ab")
    def run_ba(self):
        self.Gba = load_model(os.path.join("GAN_models", self.gba_name), custom_objects=self._objd)
        self.Gba._make_predict_function()
        print("   Gba Model loaded")
        self.run("ba")
    def spin(self):
        self.thr2.join()
    def _init_GAN(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # tf.config.gpu_options.allow_growth = True
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
        print("   _init_GAN")
        print("   Keras Loaded")
        self.gab_name = "model_gAB_insert_clean_v2_arc_a0_69"
        self.gba_name = "model_gBA_insert_clean_v2_arc_a0_69"
        self._objd = {'InstanceNormalization': InstanceNormalization}


        pass
    def gan_gen(self,imgs,ab):
        # imgs = np.expand_dims(imgs, 0)
        assert imgs.dtype==np.uint8
        # raw_shape=imgs.shape
        # if(len(raw_shape)==3):
        #     imgs=np.expand_dims(imgs,0)
        if ab=="ab":
            model=self.Gab
        else:
            model=self.Gba
        imgs=np.array(imgs).astype(np.float64) / 127.5-1.
        gen_imgs=model.predict(imgs)
        gen_imgs = (0.5 * gen_imgs + 0.5)*255
        gen_imgs=gen_imgs.astype(np.uint8)
        # print("Gened Img-> a->b")
        # if(len(raw_shape)==3):
        #     gen_imgs=np.squeeze(gen_imgs)
        return gen_imgs
if __name__=="__main__":
    gan_server=GAN_SERVER()
    gan_server.start()
    # gan_server.spin()
    # gan_server.run_ab()