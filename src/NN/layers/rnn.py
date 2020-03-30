import tensorflow as tf

from src import GlobalVariables as g
import src.mtypes as t
from .KerasLayer import KerasLayer
from .attention import SAH

K = tf.keras.backend
layers = tf.keras.layers


class LstmBlock(KerasLayer):
    def __init__(self, size: int, dropout: float = g.nn.dropout_r, return_sequences: bool = False, *args, **kwargs):
        super(LstmBlock, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.size = size
        self.dropout = dropout
        self.return_sequences = return_sequences

        self.lstm = layers.LSTM(size,
                                return_sequences=return_sequences,
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
            return_sequences=self.return_sequences
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

    def call(self, inputs, initial_state=None):
        x = self.lstm(inputs, initial_state=initial_state)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout_layer(x)

    def compute_output_shape(self, input_shape):
        return self.lstm.compute_output_shape(input_shape)


class LstmRNN(KerasLayer):
    type_size_list = t.List[int]

    def __init__(self, size_list: type_size_list, dropout: float = g.nn.dropout_r, return_sequences: bool = False,
                 use_sah=False, state_as_output=False, *args, **kwargs):
        """

        :param state_as_output:
        :param size_list:
        :param dropout:
        :param return_sequences: If True, returns the all sequence
        :param use_sah: If True, use a Self Attention Head after the first layer of LSTM
        :param args:
        :param kwargs:
        """
        super(LstmRNN, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.state_as_output = state_as_output
        self.size_list = size_list
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.use_sah = use_sah

        self.lstm_blocks = []
        self.init_lstm_blocks(size_list, dropout)

    def init_lstm_blocks(self, size_list, dropout=g.nn.dropout_r):
        for index, size in enumerate(size_list):
            return_sequence = index < len(size_list) - 1 or self.return_sequences
            self.lstm_blocks.append(LstmBlock(size=size, dropout=dropout, return_sequences=return_sequence))
            if index == 0 and self.use_sah and (len(size_list) > 1 or self.return_sequences):
                self.lstm_blocks.append(SAH(latent_size=size))

    def get_config(self):
        config = super(LstmRNN, self).get_config()
        config.update(dict(
            size_list=self.size_list,
            dropout=self.dropout,
            return_sequences=self.return_sequences,
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


class LSTMGen(KerasLayer):
    """

    """
    def __init__(self, nb_steps, dropout=g.nn.dropout_r, *args, **kwargs):
        """

        :param nb_steps:
        :param args:
        :param kwargs:
        """
        super(LSTMGen, self).__init__(*args, **kwargs)
        self.dropout = dropout
        self.nb_steps = nb_steps

        self.lstm = None
        self.nb_units = None

    def get_config(self):
        config = super(LSTMGen, self).get_config()
        config.update(
            nb_steps=self.nb_steps,
            dropout=self.dropout
        )
        return config

    def build(self, inputs_shape):
        """

        :param inputs_shape: [(batch, nb_units), (batch, nb_units), (batch, input_size)]
        :return:
        """
        state_h, state_c, x = inputs_shape
        self.nb_units = x[1]
        self.lstm = layers.LSTM(
            units=self.nb_units,
            unit_forget_bias=True,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
            return_sequences=True,
            return_state=True
        )
        self.lstm.build((x[0], 1, x[1]))

    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        state_h, state_c, x = inputs
        x = tf.expand_dims(input=x, axis=1)     # (batch, 1, size)
        outputs = []
        states = state_h, state_c
        x_ = x
        for _ in range(self.nb_steps):
            x_, state_h_, state_c_ = self.lstm(x_, initial_state=states)
            outputs.append(x_)
            states = [state_h_, state_c_]
        # outputs: List(nb_steps)[batch, nb_units)]
        stacked_outputs = tf.concat(values=outputs, axis=1)      # (batch, nb_steps, nb_units)
        return stacked_outputs

    def compute_output_shape(self, inputs_shape):
        """

        :param inputs_shape: [(batch, nb_units), (batch, nb_units), (batch, input_size)]
        :return:
        """
        state_h, state_c, x = inputs_shape
        return None, self.nb_steps, x[1]


class MultiLSTMGen(KerasLayer):
    """

    """
    def __init__(self, nb_steps, size_list, *args, dropout=g.nn.dropout_r, **kwargs):
        """

        :param size_list: List[int]
        :param nb_steps:
        :param dropout:
        :param args:
        :param dropout:
        :param kwargs:
        """
        super(MultiLSTMGen, self).__init__(*args, **kwargs)
        self.dropout = dropout
        self.nb_steps = nb_steps
        self.size_list = size_list

        self.dense_list = []
        self.first_dense = layers.Dense(units=size_list[0])     # To create first x for generation
        self.lstm_list = []
        for i in range(len(size_list)):
            if i == 0:
                # The first one need to generate the sequence from the state
                self.lstm_list.append(LSTMGen(
                    nb_steps=nb_steps,
                    dropout=dropout
                ))
                self.dense_list.append(layers.Dense(
                    units=2*size_list[0]        # To create state h and state c
                ))
            else:
                self.lstm_list.append(LstmBlock(
                    size=size_list[i],
                    dropout=dropout,
                    return_sequences=True,
                ))
                self.dense_list.append(layers.Dense(
                    units=2*size_list[i]            # To create state h and state c
                ))

    def get_config(self):
        config = super(MultiLSTMGen, self).get_config()
        config.update(
            dropout=self.dropout,
            nb_steps=self.nb_steps,
            size_list=self.size_list
        )
        return config

    def build(self, inputs_shape):
        """

        :param inputs_shape: (batch, size)
        :return:
        """
        for i in range(len(self.size_list)):
            self.dense_list[i].build(inputs_shape)
            if i == 0:
                self.first_dense.build(inputs_shape)
                self.lstm_list[0].build(
                    [(None, self.size_list[0]) for _ in range(3)]
                )
            else:
                self.lstm_list[i].build((None, self.nb_steps,  self.size_list[i-1]))

    def call(self, inputs):
        """

        :param inputs: (batch, size)
        :return:
        """
        first_x = self.first_dense(inputs)      # (batch, size)
        first_states = self.dense_list[0](inputs)
        first_states_tuple = first_states[:, :self.size_list[0]], first_states[:, self.size_list[0]:]
        sequence = self.lstm_list[0]([first_states_tuple[0], first_states_tuple[1], first_x])      # LstmGen
        # sequence: (batch, nb_steps, size)
        for i in range(1, len(self.size_list)):
            states = self.dense_list[i](inputs)     # (batch, 2*size)
            state_h, state_c = states[:, :self.size_list[i]], states[:, self.size_list:]
            # state_h: (batch, size)
            # state_c: (batch, size)
            sequence = self.lstm_list[i](sequence, initial_state=[state_h, state_c])
        return sequence

    def compute_output_shape(self, inputs_shape):
        """

        :param inputs_shape: (batch, size)
        :return:
        """
        return inputs_shape[0], self.nb_steps, self.size_list[-1]






