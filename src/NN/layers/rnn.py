import tensorflow as tf

from src import GlobalVariables as g
import src.mtypes as t
from .KerasLayer import KerasLayer
from .attention import SAH

K = tf.keras.backend
layers = tf.keras.layers


class LstmBlock(KerasLayer):
    def __init__(self, size: int, dropout: float = g.nn.dropout_r, return_sequence: bool = False, *args, **kwargs):
        super(LstmBlock, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.size = size
        self.dropout = dropout
        self.return_sequence = return_sequence

        self.lstm = layers.LSTM(size,
                                return_sequences=return_sequence,
                                unit_forget_bias=True,
                                dropout=dropout,
                                recurrent_dropout=dropout)
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout_layer = layers.Dropout(dropout)

    def get_config(self):
        config = super(LstmBlock, self).get_config()
        config.update(dict(
            size=self.size,
            dropout=self.dropout,
            return_sequence=self.return_sequence
        ))
        return config

    def build(self, input_shape):
        self.lstm.build(input_shape)
        new_shape = self.lstm.compute_output_shape(input_shape)
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout_layer.build(new_shape)
        self.set_weights_variables(self.lstm, self.batch_norm)
        super(LstmBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout_layer(x)

    def compute_output_shape(self, input_shape):
        return self.lstm.compute_output_shape(input_shape)


class LstmRNN(KerasLayer):
    type_size_list = t.List[int]

    def __init__(self, size_list: type_size_list, dropout: float = g.nn.dropout_r, return_sequence: bool = False,
                 use_sah=False, *args, **kwargs):
        """

        :param size_list:
        :param dropout:
        :param return_sequence: If True, returns the all sequence
        :param use_sah: If True, use a Self Attention Head after the first layer of LSTM
        :param args:
        :param kwargs:
        """
        super(LstmRNN, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.size_list = size_list
        self.dropout = dropout
        self.return_sequence = return_sequence
        self.use_sah = use_sah

        self.lstm_blocks = []
        self.init_lstm_blocks(size_list, dropout)

    def init_lstm_blocks(self, size_list, dropout=g.nn.dropout_r):
        for index, size in enumerate(size_list):
            return_sequence = index < len(size_list) - 1 or self.return_sequence
            self.lstm_blocks.append(LstmBlock(size=size, dropout=dropout, return_sequence=return_sequence))
            if index == 0 and self.use_sah and (len(size_list) > 1 or self.return_sequence):
                self.lstm_blocks.append(SAH(latent_size=size))

    def get_config(self):
        config = super(LstmRNN, self).get_config()
        config.update(dict(
            size_list=self.size_list,
            dropout=self.dropout,
            return_sequence=self.return_sequence,
            use_sah=self.use_sah
        ))
        return config

    def build(self, input_shape):
        new_shape = input_shape
        self.reset_weights_variables()
        for lstm in self.lstm_blocks:
            lstm.build(new_shape)
            new_shape = lstm.compute_output_shape(new_shape)
            # self.add_weights_variables(lstm)
        super(LstmRNN, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for lstm in self.lstm_blocks:
            x = lstm(x)
        return x

    def compute_output_shape(self, input_shape):
        new_shape = input_shape
        for lstm in self.lstm_blocks:
            new_shape = lstm.compute_output_shape(new_shape)
        return new_shape
