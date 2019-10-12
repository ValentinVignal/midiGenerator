import tensorflow as tf

import src.global_variables as g

K = tf.keras.backend
layers = tf.keras.layers


class LstmBlock(layers.Layer):
    def __init__(self, size, dropout=g.dropout, return_sequence=False):
        self.lstm = layers.LSTM(size,
                                return_sequences=return_sequence,
                                unit_forget_bias=True,
                                dropout=dropout,
                                recurrent_dropout=dropout)
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(dropout)
        super(LstmBlock, self).__init__()

    def build(self, input_shape):
        self.lstm.build(input_shape)
        new_shape = self.lstm.compute_output_shape(input_shape)
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout.build(new_shape)
        self._trainable_weights = self.lstm.trainable_weights + self.batch_norm.trainable_weights
        self._non_trainable_weights = self.lstm.non_trainable_weights + self.batch_norm.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(LstmBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout(x)

    def compute_output_shape(self, input_shape):
        return self.lstm.compute_output_shape(input_shape)


class LstmRNN(layers.Layer):
    def __init__(self, size_list, dropout=g.dropout):
        self.lstm_blocks = []
        self.size_list = size_list
        self.init_lstm_blocks(size_list, dropout)
        super(LstmRNN, self).__init__()

    def init_lstm_blocks(self, size_list, dropout=g.dropout):
        for index, size in enumerate(size_list):
            return_sequence = index < len(size_list) - 1
            self.lstm_blocks.append(LstmBlock(size=size, dropout=dropout, return_sequence=return_sequence))

    def build(self, input_shape):
        new_shape = input_shape
        self._trainable_weights = []
        self._non_trainable_weights = []
        for lstm in self.lstm_blocks:
            lstm.build(new_shape)
            new_shape = lstm.compute_output_shape(new_shape)
            self._trainable_weights += lstm.trainable_weights
            self._non_trainable_weights += lstm.non_trainable_weights
            # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
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
