import tensorflow as tf

from src import GlobalVariables as g
import src.mtypes as t
from .KerasLayer import KerasLayer
from .attention import SAH

K = tf.keras.backend
layers = tf.keras.layers


class Lstm(KerasLayer):
    """

    """
    def __init__(self, size: int, *args, dropout: float = g.nn.dropout_r, return_sequences: bool = True,
                 bidirectional: bool = False, **kwargs):
        """

        :param bidirectional:
        :param size:
        :param args:
        :param dropout:
        :param return_sequences:
        :param kwargs:
        """
        super(Lstm, self).__init__(*args, **kwargs)
        self.bidirectional = bidirectional
        self.size = size,
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.forward = layers.LSTM(
            units=size,
            return_sequences=return_sequences,
            dropout=dropout,
            recurrent_dropout=dropout,
            unit_forget_bias=True
        )
        self.backward = None
        if bidirectional:
            self.backward = layers.LSTM(
                units=size,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=dropout,
                unit_forget_bias=True
            )

    def get_config(self):
        config = super(Lstm, self).get_config()
        config.update(
            size=self.size,
            dropout=self.dropout,
            return_sequences=self.return_sequences,
            bidirectional=self.bidirectional
        )
        return config

    def build(self, inputs_shape):
        """

        :param inputs_shape: (batch, nb_steps, size)
        :return:
        """
        self.forward.build(inputs_shape)
        if self.bidirectionnal:
            self.backward.build(inputs_shape)

    def call(self, inputs, initial_state=None):
        """

        :param inputs: (batch, nb_steps, size)
        :param initial_state: None or:
                    if directional:
                        (batch, size) or [(batch, size)]
                    else:
                        List(2)[(batch, size)]

        :return:
        """
        initial_state_forward = None
        initial_state_backward = None
        if initial_state is not None:
            if isinstance(initial_state, list):
                initial_state_forward = initial_state[0]
            else:
                initial_state_forward = initial_state
            if self.bidirectional:
                if isinstance(initial_state, list):
                    initial_state_backward = initial_state[1]
                else:
                    initial_state_backward = initial_state
        forward = self.forward(inputs, initial_state=initial_state_forward)     # (batch, (nb_steps), size)
        backward = None
        if self.bidirectional:
            inputs_backward = inputs[:, ::-1]       # backward for time steps
            backward = self.backward(inputs_backward, initial_state=initial_state_backward)
        # (batch, (nb_steps), size)
        if self.bidirectional:
            all_ = tf.concat([forward, backward], axis=-1)
            return all_
        else:
            return forward

    def compute_output_shape(self, input_shape):
        """

        :param input_shape: (batch, nb_steps, size)
        :return:
        """
        output_shape = self.forward.compute_output_shape(input_shape)
        if self.bidirectional:
            output_shape = (*output_shape[:-1], 2 * output_shape[-1])
        return output_shape


class LstmBlock(KerasLayer):
    def __init__(self, size: int, *args, dropout: float = g.nn.dropout_r, return_sequences: bool = False,
                 bidirectional: bool = False, **kwargs):
        """

        :param size:
        :param args:
        :param dropout:
        :param return_sequences:
        :param bidirectional:
        :param kwargs:
        """
        super(LstmBlock, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.size = size
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional

        self.lstm = Lstm(
            size,
            return_sequences=return_sequences,
            unit_forget_bias=True,
            dropout=dropout,
            recurrent_dropout=dropout,
            bidirectional=bidirectional
        )
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

    def __init__(self, size_list: type_size_list, *args, dropout: float = g.nn.dropout_r, return_sequences: bool = False,
                 use_sah=False, state_as_output=False, bidirectionnal: bool = False, **kwargs):
        """

        :param state_as_output:
        :param size_list:
        :param dropout:
        :param return_sequences: If True, returns the all sequence
        :param use_sah: If True, use a Self Attention Head after the first layer of LSTM
        :param bidirectionnal:
        :param args:
        :param kwargs:
        """
        super(LstmRNN, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.bidirectionnal = bidirectionnal
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
            self.lstm_blocks.append(LstmBlock(
                size=size,
                dropout=dropout,
                return_sequences=return_sequence,
                bidirectional=self.bidirectionnal
            ))
            if index == 0 and self.use_sah and (len(size_list) > 1 or self.return_sequences):
                self.lstm_blocks.append(SAH(latent_size=size))

    def get_config(self):
        config = super(LstmRNN, self).get_config()
        config.update(dict(
            size_list=self.size_list,
            dropout=self.dropout,
            return_sequences=self.return_sequences,
            use_sah=self.use_sah,
            bidirectional=self.bidirectionnal
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
    def __init__(self, nb_steps, *args, dropout=g.nn.dropout_r, bidirectional: bool = False, **kwargs):
        """

        :param nb_steps:
        :param dropout:
        :param bidirectional:
        :param args:
        :param kwargs:
        """
        super(LSTMGen, self).__init__(*args, **kwargs)
        self.dropout = dropout
        self.nb_steps = nb_steps
        self.bidirectional = bidirectional

        self.lstm_forward = None
        self.lstm_backward = None
        self.nb_units = None

    def get_config(self):
        config = super(LSTMGen, self).get_config()
        config.update(
            nb_steps=self.nb_steps,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        return config

    def build(self, inputs_shape):
        """

        :param inputs_shape: if not bidirectional:
                                [(batch, nb_units)] or (batch, nb_units)
                            else:
                                List(2)[(batch, nb_units)]
        :return:
        """
        x = inputs_shape[0] if isinstance(inputs_shape, list) else inputs_shape

        self.nb_units = x[1]
        self.lstm_forward = layers.LSTM(
            units=self.nb_units,
            unit_forget_bias=True,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
            return_sequences=True,
            return_state=True
        )
        self.lstm_forward.build((x[0], 1, x[1]))

        if self.bidirectional:
            self.lstm_backward = layers.LSTM(
                units=self.nb_units,
                unit_forget_bias=True,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                return_sequences=True,
                return_state=True
            )
            self.lstm_backward.build((x[0], 1, x[1]))

    def call(self, inputs, initial_state=None):
        """

        :param inputs: if not bidirectional:
                            (batch, size) or [(batch, size)]
                        else:
                        (batch, size), or [(batch, size)] or List(2)[(batch, size)]
        :param initial_state:
                        if not bidirectional:
                            List(2)[(batch, size)]
                        else:
                            List(2)[List(2)[(batch, size)]]

        :return:
        """
        x_backward = None
        state_forward = None
        state_backward = None
        if self.bidirectional:
            # inputs
            if isinstance(inputs, list):
                x_forward = inputs[0]
                x_backward = inputs[1] if len(input) > 1 else inputs[0]
            else:
                x_forward = inputs
                x_backward = inputs
            # states
            if initial_state is not None:
                if isinstance(initial_state[0], list):
                    state_forward = initial_state[0]
                    state_backward = initial_state[1]
                else:
                    state_forward = initial_state
                    state_backward = initial_state
        else:
            # inputs
            x_forward = inputs[0] if isinstance(inputs, list) else inputs
            # state
            state_forward = initial_state

        x_forward = tf.expand_dims(input=x_forward, axis=1)     # (batch, 1, size)
        if self.bidirectional:
            x_backward = tf.expand_dims(x_backward, axis=1)     # (batch, 1, size)

        outputs_forward = []
        outputs_backward = []

        for _ in range(self.nb_steps):
            x_forward, state_h_forward, state_c_forward = self.lstm_forward(x_forward, initial_state=state_forward)
            outputs_forward.append(x_forward)
            state_forward = [state_h_forward, state_c_forward]
            if self.bidirectional:
                x_backward, state_h_backward, state_c_backward = self.lstm_backward(x_backward, initial_state=state_backward)
                outputs_backward.append(x_backward)
                state_backward = [state_h_backward, state_c_backward]

        # outputs_forward: List(nb_steps)[batch, 1, nb_units)]
        stacked_outputs_forward = tf.concat(values=outputs_forward, axis=1)      # (batch, nb_steps, nb_units)
        if self.bidirectional:
            stacked_outputs_backward = tf.concat(values=outputs_backward, axis=1)      # (batch, nb_steps, nb_units)
            stacked_outputs = tf.concat(
                values=[stacked_outputs_forward, stacked_outputs_backward],
                axis=-1
            )       # (batch, nb_steps, nb_units)
            return stacked_outputs

        else:
            return stacked_outputs_forward

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
    def __init__(self, nb_steps, size_list, *args, dropout=g.nn.dropout_r, bidirectional: bool =False, **kwargs):
        """

        :param size_list: List[int]
        :param nb_steps:
        :param dropout:
        :param args:
        :param dropout:
        :param bidirectional:
        :param kwargs:
        """
        super(MultiLSTMGen, self).__init__(*args, **kwargs)
        self.dropout = dropout
        self.nb_steps = nb_steps
        self.size_list = size_list
        self.bidirectional = bidirectional

        self.dense_list_forward = []
        self.dense_list_backward = [] if bidirectional else None
        self.first_dense_forward = layers.Dense(units=size_list[0])     # To create first x for generation
        self.first_dense_backward = layers.Dense(units=size_list[0]) if bidirectional else None

        self.lstm_list = []
        for i in range(len(size_list)):
            self.dense_list_forward.append(layers.Dense(
                units=2*size_list[0]        # To create state h and state c
            ))
            if bidirectional:
                self.dense_list_backward.append(layers.Dense(
                    units=2*size_list[0]        # To create state h and state c
                ))
            if i == 0:
                # The first one need to generate the sequence from the state
                self.lstm_list.append(LSTMGen(
                    nb_steps=nb_steps,
                    dropout=dropout,
                    bidirectional=bidirectional
                ))
            else:
                self.lstm_list.append(LstmBlock(
                    size=size_list[i],
                    dropout=dropout,
                    return_sequences=True,
                    bidirectional=bidirectional
                ))

    def get_config(self):
        config = super(MultiLSTMGen, self).get_config()
        config.update(
            dropout=self.dropout,
            nb_steps=self.nb_steps,
            size_list=self.size_list,
            bidirectional=self.bidirectional
        )
        return config

    def build(self, inputs_shape):
        """

        :param inputs_shape: (batch, size)
        :return:
        """
        for i in range(len(self.size_list)):
            self.dense_list_forward[i].build(inputs_shape)
            if self.bidirectional:
                self.dense_list_backward[i].build(inputs_shape)
            if i == 0:
                self.first_dense_forward.build(inputs_shape)
                if self.bidirectional:
                    self.first_dense_backward.build(inputs_shape)
                self.lstm_list[0].build((None, self.size_list[0]))
            else:
                self.lstm_list[i].build((None, self.nb_steps,  self.size_list[i-1]))

    def call(self, inputs):
        """

        :param inputs: (batch, size)
        :return:
        """
        first_x_forward = self.first_dense_forward(inputs)      # (batch, size)
        first_x_backward = self.first_dense_backward(inputs)      # (batch, size)
        first_states_forward = self.dense_list_forward[0](inputs)
        first_states_forward_list = [
            first_states_forward[:, :self.size_list[0]],
            first_states_forward[:, self.size_list[0]:]
        ]
        first_states_backward = self.dense_list_backward[0](inputs)
        first_states_backward_list = [
            first_states_backward[:, :self.size_list[0]],
            first_states_backward[:, self.size_list[0]:]
        ]
        sequence = self.lstm_list[0](
            inputs=[first_x_forward, first_x_backward],
            initial_states=[first_states_forward_list, first_states_backward_list]
        )      # LstmGen
        # sequence: (batch, nb_steps, size)
        for i in range(1, len(self.size_list)):
            states_forward = self.dense_list_forward[i](inputs)     # (batch, 2*size)
            states_backward = self.dense_list_backward[i](inputs)     # (batch, 2*size)
            state_h_forward, state_c_forward = states_forward[:, :self.size_list[i]], states_forward[:, self.size_list:]
            # state_h: (batch, size)
            # state_c: (batch, size)
            states_backward = self.dense_list_backward[i](inputs)     # (batch, 2*size)
            state_h_backward, state_c_backward = states_backward[:, :self.size_list[i]], states_backward[:, self.size_list:]
            # state_h: (batch, size)
            # state_c: (batch, size)
            sequence = self.lstm_list[i](
                inputs=sequence,
                initial_state=[
                    [state_h_forward, state_c_backward],
                    [state_h_backward, state_c_backward]
                ]
            )
        return sequence

    def compute_output_shape(self, inputs_shape):
        """

        :param inputs_shape: (batch, size)
        :return:
        """
        output_size = 2 * self.size_list[-1] if self.bidirectional else self.size_list[-1]
        return inputs_shape[0], self.nb_steps, output_size






