import tensorflow as tf

import src.NN.layers.conv_block as conv_block
import src.NN.layers.dense_coder as dense_coder
import src.global_variables as g

K = tf.keras.backend
layers = tf.keras.layers


class ConvEncoder3D(layers.Layer):
    def __init__(self, filters_list, dropout=g.dropout, time_stride=1):
        self.conv_blocks = []
        self.init_conv_blocks(filters_list, dropout=dropout, time_stride=time_stride)
        super(ConvEncoder3D, self).__init__()

    def init_conv_blocks(self, filters_list, dropout=g.dropout, time_stride=1):
        for index_list, size_list in enumerate(filters_list):
            for index, size in enumerate(size_list):
                if (index_list < len(filters_list) - 1) and (index == len(size_list) - 1):
                    strides = (1, time_stride, 2)
                else:
                    strides = (1, 1, 1)
                self.conv_block.append(conv_block.ConvBlock3D(filters=size,
                                                              strides=strides,
                                                              dropout=dropout))

    def build(self, input_shape):
        new_shape = input_shape
        self._trainable_weights = []
        self._non_trainable_weights = []
        for conv in self.conv_blocks:
            conv.build(new_shape)
            new_shape = conv.compute_output_shape(new_shape)
            self._trainable_weights += conv.trainable_weights
            self._non_trainable_weights += conv.non_trainable_weights
            # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(ConvEncoder3D, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for conv in self.conv_blocks:
            x = conv(x)
        return x

    def compute_output_shape(self, input_shape):
        new_shape = input_shape
        for conv in self.conv_blocks:
            new_shape = conv.compute_output_shape(new_shape)
        return new_shape


class Encoder3D(layers.Layer):
    def __init__(self, encoder_param, dropout=g.dropout, time_stride=1):
        self.conv_enc = ConvEncoder3D(filters_list=encoder_param['conv'],
                                   dropout=dropout,
                                   time_stride=time_stride)
        self.flatten = layers.Flatten()
        self.dense_enc = dense_coder.DenseCoder(size_list=encoder_param['dense'],
                                             dropout=dropout)
        super(Encoder3D, self).__init__()

    def build(self, input_shape):
        self.conv_enc.build(input_shape)
        new_shape = self.conv_enc.compute_output_shape(input_shape)
        self.flatten.build(new_shape)
        new_shape = self.flatten.compute_output_shape(new_shape)
        self.dense_enc.build(new_shape)
        self._trainable_weigths = self.conv_enc.trainable_weights + self.dense_enc.trainable_weights
        self._non_trainable_weigths = self.conv_enc.non_trainable_weights + self.dense_enc.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(Encoder3D, self).build(input_shape)

    def call(self, inputs):
        x = self.conv_enc(inputs)
        x = self.flatten(x)
        x = self.dense_enc(x)
        return x

    def compute_output_shape(self, input_shape):
        new_shape = self.conv_enc.compute_output_shape(input_shape)
        new_shape = self.flatten.compute_output_shape(new_shape)
        new_shape = self.dense_enc.compute_output_shape(new_shape)
        return new_shape
