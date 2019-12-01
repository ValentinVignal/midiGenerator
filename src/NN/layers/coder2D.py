import tensorflow as tf

from . import conv as conv
from . import dense as dense
import src.global_variables as g
import src.mtypes as t
from .KerasLayer import KerasLayer

K = tf.keras.backend
layers = tf.keras.layers


class ConvEncoder2D(KerasLayer):
    type_filters_list = t.List[t.List[int]]

    def __init__(self, filters_list: type_filters_list, dropout: float = g.dropout, time_stride: int = 1,
                 last_pool: bool = False):
        """
        :param filters_list: List[List[int]]:
        :param dropout: float:
        :param time_stride: int:
        :type last_pool: bool:
        """
        self.last_pool = last_pool
        self.conv_blocks = []
        self.init_conv_blocks(filters_list, dropout=dropout, time_stride=time_stride)
        super(ConvEncoder2D, self).__init__()

    def init_conv_blocks(self, filters_list: type_filters_list, dropout: float = g.dropout, time_stride: int = 1):
        for index_list, size_list in enumerate(filters_list):
            for index, size in enumerate(size_list):
                if index == len(size_list) - 1:
                    if (index_list < len(filters_list) - 1) or self.last_pool:
                        strides = (time_stride, 2)
                    else:
                        strides = (1, 1)
                else:
                    strides = (1, 1)
                self.conv_blocks.append(conv.ConvBlock2D(filters=size,
                                                         strides=strides,
                                                         dropout=dropout))

    def build(self, input_shape):
        new_shape = input_shape
        self.reset_weights_variables()
        for conv in self.conv_blocks:
            conv.build(new_shape)
            new_shape = conv.compute_output_shape(new_shape)
            self.add_weights_variables(conv)
        super(ConvEncoder2D, self).build(input_shape)

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


class Encoder2D(KerasLayer):
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

        self.conv_enc = ConvEncoder2D(filters_list=encoder_param['conv'],
                                      dropout=dropout,
                                      time_stride=time_stride,
                                      last_pool=self.last_pool)
        if time_distributed:
            self.flatten = layers.TimeDistributed(layers.Flatten())
        else:
            self.flatten = layers.Flatten()
        self.dense_enc = dense.DenseCoder(size_list=encoder_param['dense'],
                                          dropout=dropout)
        super(Encoder2D, self).__init__()

    def build(self, input_shape):
        self.conv_enc.build(input_shape)
        new_shape = self.conv_enc.compute_output_shape(input_shape)
        self.flatten.build(new_shape)
        new_shape = self.flatten.compute_output_shape(new_shape)
        self.dense_enc.build(new_shape)
        self.set_weights_variables(self.conv_enc, self.dense_enc)
        super(Encoder2D, self).build(input_shape)

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


class ConvDecoder2D(KerasLayer):
    type_filters_list = t.List[t.List[int]]
    type_shapes_after_upsize = t.Optional[t.List[t.bshape]]

    nb_instance = 0

    def __init__(self, filters_list: type_filters_list, dropout: float = g.dropout, time_stride: int = 1,
                 shapes_after_upsize: type_shapes_after_upsize = None, first_pool: bool = False):
        """

        :param filters_list: List[List[int]]:
        :param dropout: float:
        :param time_stride: int:
        :param shapes_after_upsize: Optional[List[bshape]]: The shapes after a pool (even the first one if first_pool = True)
        :param first_pool: bool:
        """
        super(ConvDecoder2D, self).__init__()
        self.shapes_after_upsize = shapes_after_upsize
        self.first_pool = first_pool

        self.conv_blocks = []
        self.init_conv_blocks(filters_list, dropout=dropout, time_stride=time_stride, final_shapes=shapes_after_upsize)

    def init_conv_blocks(self, filters_list: type_filters_list, dropout: float = g.dropout, time_stride: int = 1,
                         final_shapes: type_shapes_after_upsize = None, first_pool: bool = False):
        for index_list, size_list in enumerate(filters_list):
            for index, size in enumerate(size_list):
                if index == 0:
                    if index_list > 0 or first_pool:
                        strides = (time_stride, 2)
                        final_shape = final_shapes[index_list]
                    else:
                        strides = (1, 1)
                        final_shape = None
                else:
                    strides = (1, 1)
                    final_shape = None
                self.conv_blocks.append(conv.ConvTransposedBlock2D(filters=size,
                                                                   strides=strides,
                                                                   dropout=dropout,
                                                                   final_shape=final_shape))

    def build(self, input_shape):
        new_shape = input_shape
        self.reset_weights_variables()
        print('ConvDecoder2D, build, input_shape', input_shape)
        for conv in self.conv_blocks:
            print('ConvDecoder2D, build, new_shape', new_shape)
            conv.build(new_shape)
            new_shape = conv.compute_output_shape(new_shape)
            self.add_weights_variables(conv)
        super(ConvDecoder2D, self).build(input_shape)

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


class Decoder2D(KerasLayer):
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
        super(Decoder2D, self).__init__()
        self.decoder_param = decoder_param
        self.shapes_after_upsize = shapes_after_upsize
        self.first_upsize = first_upsize
        self.shape_before_conv = shape_before_conv

        self.dense_dec = dense.DenseCoder(size_list=decoder_param['dense'],
                                          dropout=dropout)
        print('Decoder2D, init, decoder_param[dense]', decoder_param['dense'])
        print('Decoder2D, init, shape before conv', shape_before_conv)
        self.reshape = layers.Reshape(self.shape_before_conv)
        self.conv_dec = ConvDecoder2D(filters_list=decoder_param['conv'],
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
        self.dense_dec.build(input_shape)
        new_shape = self.dense_dec.compute_output_shape(input_shape)
        self.reshape.build(new_shape)
        new_shape = self.reshape.compute_output_shape(new_shape)
        print('Decoder2D, build, reshaped', new_shape)
        self.conv_dec.build(new_shape)

        self.set_weights_variables(self.dense_dec, self.conv_dec)
        super(Decoder2D, self).build(input_shape)

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
