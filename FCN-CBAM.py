# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import *
# from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, Cropping2D
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import sgydata
import os
import scipy.io as sio

import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
# np.random.seed(5)
# encoding:utf-8
os.environ["CUDA_VISIBLE_DEVICES"] = " 0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from Attention128 import CBAM128
from Attention64 import CBAM64
np.random.seed(5)
class FCNloca(object):
    def __init__(self, img_rows=12, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols
    def load_data(self):
        #xyzrange = [[0.190, 0.0035, 108], [-0.220, 0.0034, 128], [3.010, 0.0037, 64]]
        #xyzrange = [[0.190, 0.0035, 108], [3.010, 0.0037, 64]]
        # xyzrange=[[0.15,0.005,72],[-0.2,0.006,64],[3.05,0.006,30]]
        wave_train, loca_train = sgydata.load_sgylist_xyz1(sgylist=['./train1/', 'train1_new.txt'],
                                                           sgyr=[0, -1, 1], x=[0.190,0.0035,144],y=[-0.200,0.0035,144],z=[3.000, 0.0035, 144],  r=(0.035) ** 2,
                                                           shuffle='true', shiftdata=[list(range(-15, 5)), 1])
        print(loca_train.shape)
        # shiftdata=[list(range(20,50))+list(range(-200,-20)),1])
        loca_train = np.reshape(loca_train, (len(loca_train), 144, 1, 3))
        print('end load_data()')
        # sio.savemat('label.mat', {'xyzrange':xyzrange,"data":wave_train,"ydata":loca_train})
        return wave_train, loca_train

    def get_network(self):
        inputs = Input((12, 512, 3))

        conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            inputs)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        a = CBAM64(in_channels=64)(conv1)
        pool1 = MaxPooling2D(pool_size=(1, 4))(a)  # 12 128 64

        print("pool1 shape:", pool1.shape)
        conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            pool1)
        # print("conv2 shape:",conv2.shape)
        conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            conv2)
        print("conv2 shape:", conv2.shape)
        yy = CBAM128(in_channels=128)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 4))(yy)

        print("pool2 shape:", pool2.shape)
        conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 4))(conv3)
        # y = CBAM(in_channels=256)(pool3)
        print("pool3 shape:", pool3.shape)
        conv4 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            pool3)
        conv4 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            conv4)
        drop4 = Dropout(0.5)(conv4)  # 3 8 512
        print(drop4.shape)
        #x = CBAM3(in_channels=512)(drop4)
        pool4 = MaxPooling2D(pool_size=(1, 8))(drop4)
        #

        print("pool4 shape:", pool4.shape)
        conv5 = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            pool4)
        conv5 = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            conv5)
        drop5 = Dropout(0.5)(conv5)
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(3, 1))(drop5))
        #    merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
        conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(4, 1))(conv6))
        #    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
        conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            conv7)
        conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(4, 1))(conv7))
        conv8 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv9 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv10 = Conv2D(3, kernel_size=(5, 5), activation='sigmoid', padding='same')(conv9)
        model = Model(inputs=inputs, outputs=conv10)
        print('conv10:', conv10.shape)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        #    model.compile(optimizer = 'sgd', loss = losses.mean_squared_error)
        return model

    def train(self):
        print("loading data  ")
        wave_train, loca_train = self.load_data()
        print("loading data done")
        model = self.get_network()
        model.summary()  # 输出模型各层的参数状�?
        print("got network")

        model_checkpoint = ModelCheckpoint('FCN_zhuyili1.hdf5', monitor='val_loss', verbose=1,
                                           save_best_only=True)  # 'FCNloca.hdf5'保存模型的路径；monitor需要监视的�?
        print('Fitting model...')
        hist = model.fit(wave_train, loca_train, batch_size=4, epochs=100, verbose=1,
                         shuffle=True, validation_split=0.1,
                         callbacks=[model_checkpoint])  # model.fit返回的就是history类型;history中有很多数据
        # print(hist.history['acc'])
        f = open('FCN_zhuyili1.txt', 'w')
        f.write('{} {} {}\n'.format(hist.epoch, hist.history['loss'], hist.history['val_loss']))

        f.close()


if __name__ == '__main__':
    fcnloca = FCNloca()
    fcnloca.train()

# verbose：信息展示模式，0�?。为1表示输出epoch模型保存信息，默认为0表示不输出该信息，信息形如：
#   Epoch 00001: val_acc improved from -inf to 0.49240, saving model to /xxx/checkpoint/model_001-0.3902.h5







