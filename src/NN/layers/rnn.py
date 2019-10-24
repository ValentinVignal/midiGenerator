import tensorflow as tf

import src.global_variables as g
import src.mtypes as t
from .layers import KerasLayer

K = tf.keras.backend
layers = tf.keras.layers


class LstmBlock(KerasLayer):
    def __init__(self, size: int, dropout: float = g.dropout, return_sequence: bool = False):
        super(LstmBlock, self).__init__()
        self.lstm = layers.LSTM(size,
                                return_sequences=return_sequence,
                                unit_forget_bias=True,
                                dropout=dropout,
                                recurrent_dropout=dropout)
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        self.lstm.build(input_shape)
        new_shape = self.lstm.compute_output_shape(input_shape)
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout.build(new_shape)
        self.set_weights_variables(self.lstm, self.batch_norm)
        super(LstmBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout(x)

    def compute_output_shape(self, input_shape):
        return self.lstm.compute_output_shape(input_shape)


class LstmRNN(KerasLayer):

    type_size_list = t.List[int]

    def __init__(self, size_list: type_size_list, dropout: float = g.dropout):
        super(LstmRNN, self).__init__()
        self.lstm_blocks = []
        self.size_list = size_list
        self.init_lstm_blocks(size_list, dropout)

    def init_lstm_blocks(self, size_list, dropout=g.dropout):
        for index, size in enumerate(size_list):
            return_sequence = index < len(size_list) - 1
            self.lstm_blocks.append(LstmBlock(size=size, dropout=dropout, return_sequence=return_sequence))

    def build(self, input_shape):
        new_shape = input_shape
        self.reset_weights_variables()
        for lstm in self.lstm_blocks:
            lstm.build(new_shape)
            new_shape = lstm.compute_output_shape(new_shape)
            self.add_weights_variables(lstm)
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
