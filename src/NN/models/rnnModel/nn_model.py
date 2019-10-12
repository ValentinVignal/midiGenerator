import tensorflow as tf
import warnings

import src.eval_string as es
import src.NN.losses as l
import src.global_variables as g
import src.NN.layers as mlayers

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda
K = tf.keras.backend

"""

LSTM with encoder convolutional layers and the deconder use transposed convolutional layer

"""


def create_model(input_param, model_param, nb_steps, step_length, optimizer, type_loss=g.type_loss,
                 model_options={}
                 ):
    """

    :param input_param:
    :param nb_steps:
    :param model_param:
    :param step_length:
    :param optimizer:
    :param type_loss:
    :param model_options:
    :return: the neural network:
    """

    # ---------- Model options ----------
    mmodel_options = {
        'dropout': g.dropout,
        'lambdas_loss': g.lambdas_loss,
    }
    mmodel_options.update(model_options)

    dropout = mmodel_options['dropout']

    lambda_loss_activation, lambda_loss_duration = g.get_lambdas_loss(mmodel_options['lambdas_loss'])

    print('Definition of the graph ...')

    # --------- End model options ----------

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    env = {
        'nb_instruments': nb_instruments,
        'input_size': input_size,
        'nb_steps': nb_steps
    }

    if step_length == 4 or step_length == 1:
        # It is working either on 'Beat' or on 'Note'
        # We don't want to stride on the time axis
        time_stride = 1
    elif step_length == 16:
        # It is working on 'Measure'
        # We want to stride on the time axis (the size is considered as big enough)
        time_stride = 2
    else:
        warnings.warn(
            f'The model is not designed wo work with a step_length {step_length} not included in (8, 16),' +
            'some errors might occur',
            Warning)
        time_stride = 1 if step_length < 16 else 2

    midi_shape = (nb_steps, step_length, input_size, 1)  # (batch, step_length, nb_step, input_size)

    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------

    class MyModel(tf.keras.Model):

        def __init__(self, input_shape, model_param, **kwargs):
            super(MyModel, self).__init__(name='MyModel', **kwargs)
            self.model_param_enc = model_param
            self.model_param_dec = MyModel.compute_model_param_dec(input_shape, self.model_param_enc)
            self.final_shapes = MyModel.compute_final_shapes(input_shape=input_shape,
                                                             model_param_conv=self.model_param_enc['conv'])
            print('final shapes', self.final_shapes)
            self.encoder = mlayers.coder3D.Encoder3D(encoder_param=self.model_param_enc,
                                                     dropout=dropout,
                                                     time_stride=time_stride)
            self.rnn = mlayers.rnn.LstmRNN(
                model_param['lstm'])  # TODO : Put eval in it and declare nb_instrument blablabla...
            self.decoder = mlayers.coder3D.Decoder3D(decoder_param=self.model_param_dec,
                                                     dropout=dropout,
                                                     time_stride=time_stride,
                                                     final_shapes=self.final_shapes)
            self.last_layer = mlayers.last.LastMono(softmax_axis=2)

        @staticmethod
        def compute_final_shapes(input_shape, model_param_conv):
            return mlayers.conv.new_shapes_conv(
                input_shape=(*input_shape[:-1], nb_instruments),
                strides_list=[(1, time_stride, 2) for i in range(len(model_param_conv))],
                filters_list=[l[-1] for l in model_param_conv]
            )[::-1]

        @staticmethod
        def compute_model_param_dec(input_shape, model_param_enc):
            conv_enc = model_param_enc['conv']
            dense_enc = model_param['dense']

            # --- Compute the last size of the convolution (so we can add a dense layer of this size in the decoder ---
            nb_pool = len(conv_enc) - 1  # nb_times there is a stride == 2 in the conv encoder
            # 1. compute the last shape
            last_shapes_conv_enc = mlayers.conv.new_shapes_conv(input_shape=(*input_shape[:-1], nb_instruments),
                                                               strides_list=[(1, time_stride, 2) for i in range(nb_pool)],
                                                               filters_list=[conv_enc[-1][-1] for i in range(nb_pool)])
            print('last shapes conv enc', last_shapes_conv_enc)
            last_shape_conv_enc = last_shapes_conv_enc[-1]
            # 2. compute the last size
            last_size_conv_enc = 1
            for i, s in enumerate(last_shape_conv_enc[1:]):
                last_size_conv_enc *= s
            print('l')

            # --- Create the dictionnary to return ---
            model_param_dec = dict(
                dense=dense_enc[::-1] + [last_size_conv_enc],
                conv=mlayers.conv.reverse_conv_param(original_dim=nb_instruments, param_list=conv_enc)
            )
            return model_param_dec

        def build(self, input_shape):
            print('input shape', input_shape)
            new_shape = (*input_shape[0][:-1], input_shape[0][-1] * len(input_shape))
            print('new shape 1', new_shape)
            self.encoder.build(new_shape)
            new_shape = self.encoder.compute_output_shape(new_shape)
            print('new shape 2', new_shape)
            self.rnn.build(new_shape)
            new_shape = self.rnn.compute_output_shape(new_shape)
            self.decoder.build(new_shape)
            new_shape = self.decoder.compute_output_shape(new_shape)
            self.last_layer.build(new_shape)
            super(MyModel, self).build(input_shape)

        def call(self, inputs):
            x = layers.concatenate(inputs, axis=4)  # (batch, nb_steps, step_length, input_size, nb_instruments)
            x = self.encoder(x)
            x = self.rnn(x)
            x = self.decoder(x)
            x = self.last_layer(x, names='Output_')
            return x

    model = MyModel(input_shape=midi_shape, model_param=model_param)
    model.build([(None, *midi_shape) for inst in range(nb_instruments)])

    # ------------------ Losses -----------------
    # Define losses dict
    losses = {}
    for inst in range(nb_instruments):
        losses[f'Output_{inst}'] = l.loss_function_mono

    model.compile(loss='mae', optimizer=optimizer, metrics=[l.acc_mono])

    return model, losses, (lambda_loss_activation, lambda_loss_duration)
