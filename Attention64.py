# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Layer, Conv2D, Activation, Concatenate,Reshape, Add, Multiply
class ChannelAttention(Layer):
    def __init__(self, in_channels=64, ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.fc1 = Conv2D(in_channels // ratio, kernel_size=(1, 1), padding='same', use_bias=False)
        self.relu = Activation('relu')
        self.fc2 = Conv2D(in_channels, kernel_size=(1, 1), padding='same', use_bias=False)
        self.sigmoid = Activation('sigmoid')
    def call(self, inputs):
        x_avg = self.avg_pool(inputs)
        x_avg = Reshape((1, 1, inputs.shape[-1]))(x_avg)
        x_max = self.max_pool(inputs)
        x_max = Reshape((1, 1, inputs.shape[-1]))(x_max)
        x_avg_out = self.fc2(self.relu(self.fc1(x_avg)))
        x_max_out = self.fc2(self.relu(self.fc1(x_max)))
        out = Add()([x_avg_out, x_max_out])
        out = self.sigmoid(out)
        return Multiply()([inputs, out])
    def get_config(self):
        config = super().get_config()
        return config
#### Spatial Attention
class SpatialAttention(Layer):
    def __init__(self, kernel_size=3,**kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv1 = Conv2D(1, kernel_size=kernel_size, padding='same', use_bias=False)
        self.sigmoid = Activation('sigmoid')
    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=-1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return Multiply()([inputs, x])
    def get_config(self):
        config = super().get_config()
        return config
#### CBAM
class CBAM64(Layer):
    def __init__(self, in_channels=64, ratio=16, kernel_size=3,**kwargs):
        super(CBAM64, self).__init__(**kwargs)
        self.channel_attention = ChannelAttention(in_channels, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x
    def get_config(self):
        config = super().get_config()
        return config
class SpatialAttention1(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention1, self).__init__(**kwargs)
        self.conv1 = Conv2D(filters=1, kernel_size=7, padding='same', use_bias=False)
        self.sigmoid = Activation('sigmoid')
    def call(self, inputs):
        # 压缩通道提取空间信息
        max_out = MaxPooling2D(pool_size=(inputs.shape[-3], inputs.shape[-2]), strides=1, padding='same')(inputs)
        avg_out = AveragePooling2D(pool_size=(inputs.shape[-3], inputs.shape[-2]), strides=1, padding='same')(inputs)
        # 经过卷积提取空间注意力权
        concatenated = Concatenate(axis=-1)([max_out, avg_out])
        out = self.conv1(concatenated)
        # 输出非负
        out = self.sigmoid(out)
        return out
    def get_config(self):
        config = super().get_config()
        return config

