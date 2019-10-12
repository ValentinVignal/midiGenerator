import tensorflow as tf

import src.NN.layers as mlayers
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
                self.conv_blocks.append(mlayers.conv.ConvBlock3D(filters=size,
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
    def __init__(self, encoder_param, dropout=g.dropout, time_stride=1, time_distributed=True):
        self.conv_enc = ConvEncoder3D(filters_list=encoder_param['conv'],
                                      dropout=dropout,
                                      time_stride=time_stride)
        if time_distributed:
            self.flatten = layers.TimeDistributed(layers.Flatten())
        else:
            self.flattn = layers.Flatten
        self.dense_enc = mlayers.dense.DenseCoder(size_list=encoder_param['dense'],
                                                  dropout=dropout)
        super(Encoder3D, self).__init__()

    def build(self, input_shape):
        self.conv_enc.build(input_shape)
        new_shape = self.conv_enc.compute_output_shape(input_shape)
        self.flatten.build(new_shape)
        new_shape = self.flatten.compute_output_shape(new_shape)
        self.dense_enc.build(new_shape)
        self._trainable_weights = self.conv_enc.trainable_weights + self.dense_enc.trainable_weights
        self._non_trainable_weights = self.conv_enc.non_trainable_weights + self.dense_enc.non_trainable_weights
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


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ----------------------------------------------------------------------------------------------------
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class ConvDecoder3D(layers.Layer):
    def __init__(self, filters_list, dropout=g.dropout, time_stride=1, final_shapes=None):
        self.conv_blocks = []
        self.final_shapes = final_shapes
        self.init_conv_blocks(filters_list, dropout=dropout, time_stride=time_stride, final_shapes=final_shapes)
        super(ConvDecoder3D, self).__init__()

    def init_conv_blocks(self, filters_list, dropout=g.dropout, time_stride=1, final_shapes=None):
        for index_list, size_list in enumerate(filters_list):
            for index, size in enumerate(size_list):
                if (index_list > 0) and (index == 0):
                    strides = (1, time_stride, 2)
                    final_shape = final_shapes[index_list]
                else:
                    strides = (1, 1, 1)
                    final_shape = None
                self.conv_blocks.append(mlayers.conv.ConvTransposedBlock3D(filters=size,
                                                                           strides=strides,
                                                                           dropout=dropout,
                                                                           final_shape=final_shape))

    def build(self, input_shape):
        new_shape = input_shape
        self._trainable_weights = []
        self._non_trainable_weights = []
        for conv in self.conv_blocks:
            print('new_shape in build ConvDecoder3D', new_shape)
            conv.build(new_shape)
            new_shape = conv.compute_output_shape(new_shape)
            self._trainable_weights += conv.trainable_weights
            self._non_trainable_weights += conv.non_trainable_weights
            # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(ConvDecoder3D, self).build(input_shape)

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


class Decoder3D(layers.Layer):
    def __init__(self, decoder_param, dropout=g.dropout, time_stride=1, final_shapes=None):
        print('decoder param', decoder_param)
        self.final_shapes = final_shapes
        self.shape_before_conv = (1, *self.final_shapes[1][1:-1], self.final_shapes[0][-1])
        self.dense_dec = mlayers.dense.DenseCoder(size_list=decoder_param['dense'],
                                                  dropout=dropout)
        self.reshape = layers.Reshape(self.shape_before_conv)
        self.conv_dec = ConvDecoder3D(filters_list=decoder_param['conv'],
                                      dropout=dropout,
                                      time_stride=time_stride,
                                      final_shapes=final_shapes)
        super(Decoder3D, self).__init__()

    def build(self, input_shape):
        print('decoder input shape', input_shape)
        self.dense_dec.build(input_shape)
        new_shape = self.dense_dec.compute_output_shape(input_shape)
        print('shape before conv', self.shape_before_conv)
        print('new_shape before reshape', new_shape)
        self.reshape.build(new_shape)
        new_shape = self.reshape.compute_output_shape(new_shape)
        print('new shape after reshape', new_shape)
        self.conv_dec.build(new_shape)

        self._trainable_weights = self.dense_dec.trainable_weights + self.conv_dec.trainable_weights
        self._non_trainable_weights = self.dense_dec.non_trainable_weights + self.conv_dec.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(Decoder3D, self).build(input_shape)

    def call(self, inputs):
        x = self.dense_dec(inputs)
        x = self.reshape(x)
        x = self.conv_dec(x)
        return x

    def compute_output_shape(self, input_shape):
        new_shape = self.dense_dec.compute_output_shape(input_shape)
        new_shape = self.reshape.compute_output_shape(new_shape)
        new_shape = self.conv_dec.compute_output_shape(new_shape)
        return new_shape
