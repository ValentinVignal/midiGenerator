import tensorflow as tf

import src.NN.layers as mlayers
import src.global_variables as g
import src.mtypes as t

K = tf.keras.backend
layers = tf.keras.layers


class ConvEncoder3D(layers.Layer):
    type_filters_list = t.List[t.List[int]]

    def __init__(self, filters_list: type_filters_list, dropout: float = g.dropout, time_stride: int = 1,
                 last_pool: bool = False):
        """

        :param filters_list: List[List[int]]:
        :param dropout: float:
        :param time_stride: int:
        :type last_pool: bool:
        """
        super(ConvEncoder3D, self).__init__()
        self.last_pool = last_pool
        self.conv_blocks = []
        self.init_conv_blocks(filters_list, dropout=dropout, time_stride=time_stride)

    def init_conv_blocks(self, filters_list: type_filters_list, dropout: float = g.dropout, time_stride: int = 1):
        for index_list, size_list in enumerate(filters_list):
            for index, size in enumerate(size_list):
                if index == len(size_list) - 1:
                    if (index_list < len(filters_list) - 1) or self.last_pool:
                        strides = (1, time_stride, 2)
                    else:
                        strides = (1, 1, 1)
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
    type_encoder_param = t.Dict[str, t.Union[
        t.List[t.List[int]],  # conv
        t.List[int]  # dense
    ]]

    def __init__(self, encoder_param: type_encoder_param, dropout: float = g.dropout, time_stride: int = 1,
                 time_distributed: bool = True, last_pool: bool = False):
        """

        :param encoder_param: {
            'conv': [[4, 8], [16, 8], [4, 8]],
            'dense': [24, 12]
        }
        :param dropout: float:
        :param time_stride: int:
        :param time_distributed: bool:
        :param last_pool: bool:
        """
        self.last_pool = last_pool

        self.conv_enc = ConvEncoder3D(filters_list=encoder_param['conv'],
                                      dropout=dropout,
                                      time_stride=time_stride,
                                      last_pool=self.last_pool)
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
    type_filters_list = t.List[t.List[int]]
    type_shapes_after_upsize = t.Optional[t.List[t.bshape]]

    def __init__(self, filters_list: type_filters_list, dropout: float = g.dropout, time_stride: int = 1,
                 shapes_after_upsize: type_shapes_after_upsize = None, first_pool: bool = False):
        """

        :param filters_list: List[List[int]]:
        :param dropout: float:
        :param time_stride: int:
        :param shapes_after_upsize: Optional[List[bshape]]: The shapes after a pool (even the first one if first_pool = True)
        :param first_pool: bool:
        """
        self.shapes_after_upsize = shapes_after_upsize
        self.first_pool = first_pool

        self.conv_blocks = []
        self.init_conv_blocks(filters_list, dropout=dropout, time_stride=time_stride, final_shapes=shapes_after_upsize)
        super(ConvDecoder3D, self).__init__()

    def init_conv_blocks(self, filters_list: type_filters_list, dropout: float = g.dropout, time_stride: int = 1,
                         final_shapes: type_shapes_after_upsize = None, first_pool: bool = False):
        print('ConvDecoder3D init_conv_blocks : filters_list', filters_list, 'shapes_after_upsize', final_shapes)
        for index_list, size_list in enumerate(filters_list):
            for index, size in enumerate(size_list):
                if index == 0:
                    if index_list > 0 or first_pool:
                        strides = (1, time_stride, 2)
                        final_shape = final_shapes[index_list]
                    else:
                        strides = (1, 1, 1)
                        final_shape = None
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
            print('ConvDecoder3D build new_shape', new_shape)
            conv.build(new_shape)
            new_shape = conv.compute_output_shape(new_shape)
            self._trainable_weights += conv.trainable_weights
            self._non_trainable_weights += conv.non_trainable_weights
            # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(ConvDecoder3D, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for index, conv in enumerate(self.conv_blocks):
            x = conv(x)
        return x

    def compute_output_shape(self, input_shape):
        new_shape = input_shape
        for conv in self.conv_blocks:
            new_shape = conv.compute_output_shape(new_shape)
        return new_shape


class Decoder3D(layers.Layer):
    type_decoder_param_conv = t.List[t.List[int]]
    type_decoder_param = t.Dict[str, t.Union[
        type_decoder_param_conv,  # conv
        t.List[int],  # dense,
    ]]
    type_shapes_after_upsize = t.Optional[t.List[t.bshape]]

    def __init__(self, decoder_param: type_decoder_param, shape_before_conv: t.shape, dropout: float = g.dropout,
                 time_stride: int = 1, shapes_after_upsize: type_shapes_after_upsize = None,
                 first_upsize: bool = False):
        """

        :param decoder_param: {
            'conv': [[4, 8], [16, 8], [4, 8]],
            'dense': [24, 12]
        }
        :param dropout: float:
        :param time_stride: int:
        :param shapes_after_upsize: Optional[List[bshape]]:
            ⚠ batch dim IS IN the shape ⚠
        :param first_upsize: bool:
        """
        super(Decoder3D, self).__init__()
        print('Decoder3D decoder_param', decoder_param)
        print('Decoder3D shapes_after_upsize', shapes_after_upsize)
        self.decoder_param = decoder_param
        self.shapes_after_upsize = shapes_after_upsize
        self.first_upsize = first_upsize
        self.shape_before_conv = shape_before_conv

        # self.shape_before_conv: t.shape = Decoder3D.compute_shape_before_conv(
        #     last_filter=self.decoder_param['conv'][0][0],
        #     shapes_after_upsize=self.shapes_after_upsize,
        #     first_upsize=self.first_upsize)
        print('Decoder3D shape before conv:', self.shape_before_conv)

        self.dense_dec = mlayers.dense.DenseCoder(size_list=decoder_param['dense'],
                                                  dropout=dropout)
        self.reshape = layers.Reshape(self.shape_before_conv)
        self.conv_dec = ConvDecoder3D(filters_list=decoder_param['conv'],
                                      dropout=dropout,
                                      time_stride=time_stride,
                                      shapes_after_upsize=shapes_after_upsize,
                                      first_pool=first_upsize)

    @staticmethod
    def compute_shape_before_conv(last_filter: int, shapes_after_upsize: type_shapes_after_upsize,
                                  first_upsize: bool) -> t.shape:
        ind = 0 if first_upsize else 1
        return (1, *shapes_after_upsize[ind][2:-1], last_filter)

    def build(self, input_shape):
        print('Decoder3D build : input shape', input_shape)
        self.dense_dec.build(input_shape)
        new_shape = self.dense_dec.compute_output_shape(input_shape)
        print('Decoder3D build : shape before conv', self.shape_before_conv)
        print('Decoder3D build : new_shape before reshape', new_shape)
        self.reshape.build(new_shape)
        new_shape = self.reshape.compute_output_shape(new_shape)
        print('Decoder3D build : new shape after reshape', new_shape)
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
