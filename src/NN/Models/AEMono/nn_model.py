import tensorflow as tf
import numpy as np
import argparse
import json
import sys

if __name__ == '__main__':
    sys.path.append(sys.path[0] + '\\..\\..\\..\\..')

import src.NN.losses as mlosses
import src.NN.metrics as mmetrics
import src.global_variables as g
import src.NN.layers as mlayers
from src.NN.Models.KerasModel import KerasModel
import src.NN.shapes.convolution as s_conv
import src.NN.shapes.time as s_time
from src.eval_string import eval_object

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda
K = tf.keras.backend


def create_model(input_param, model_param, nb_steps, step_length, optimizer, type_loss=g.type_loss,
                 model_options={}
                 ):
    """

    :param input_param: {
                            nb_instruments,
                            input_size
                        }
                        (Comes from the dataset)
    :param model_param: (Comes from the .json)
    :param nb_steps: (Comes from the user)
    :param step_length: (Comes from the user)
    :param optimizer:
    :param type_loss: (Comes from the user)
    :param model_options: (Comes from the user)
    :return: the neural network:
    """

    # ---------- Model options ----------
    mmodel_options = {
        'dropout': g.dropout,
        'lambdas_loss': g.lambdas_loss,
        'sample': g.sample
    }
    mmodel_options.update(model_options)

    dropout = mmodel_options['dropout']

    lambda_loss_activation, lambda_loss_duration = g.get_lambdas_loss(mmodel_options['lambdas_loss'])

    print('Definition of the graph ...')

    # --------- End model options ----------

    # --------- Variables ----------

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    env = {
        'nb_instruments': nb_instruments,
        'input_size': input_size,
        'nb_steps': nb_steps
    }

    model_param = eval_object(model_param, env=env)

    time_stride = s_time.time_stride(step_length)

    midi_shape = (nb_steps, step_length, input_size, 1)  # (batch, nb_steps, step_length, input_size, 2)

    # ----- For the decoder -----

    model_param_dec, shape_before_conv_dec = s_conv.compute_model_param_dec(
        input_shape=(*midi_shape[:-1], nb_instruments),
        model_param_enc=model_param,
        strides=[(time_stride, 2) for i in range(len(model_param['conv']) - 1)]
    )

    shapes_before_pooling = s_conv.compute_shapes_before_pooling(
        input_shape=midi_shape,
        model_param_conv=model_param['conv'],
        strides=(time_stride, 2)
    )
    # Put time to 1 :
    shapes_before_pooling = s_time.time_step_to_x(l=shapes_before_pooling, axis=1, x=1)

    # --------- End Variables ----------

    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------

    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, step_length, input_size, 2)]

    inputs_inst_step = [mlayers.shapes.Unstack(axis=0)(input_inst) for input_inst in
                        inputs_midi]  # List(nb_instruments, nb_steps)[(batch, step_length, channels)]

    # ------------------------------ Encoding ------------------------------

    encoders = [mlayers.coder2D.Encoder2D(
        encoder_param=model_param,
        dropout=dropout,
        time_stride=time_stride
    ) for inst in range(nb_instruments)]

    encoded_step_inst = [[encoders[inst](inputs_inst_step[inst][step]) for inst in range(nb_instruments)] for step in
                         range(nb_steps)]  # List(nb_inst, nb_instruments)[(batch, size)]

    encoded_steps = [layers.concatenate(encoded_step_inst[step]) for step in
                     range(nb_steps)]  # List(nb_steps)[(batch, size)]
    encoded = mlayers.shapes.Stack(axis=0)(encoded_steps)

    # ------------------------------ RNN ------------------------------

    rnn_output = mlayers.rnn.LstmRNN(
        model_param['lstm'])(encoded)

    # ------------------------------ Decoding ------------------------------

    decoders = [mlayers.coder2D.Decoder2D(
        decoder_param=model_param_dec,
        shape_before_conv=shape_before_conv_dec,
        dropout=dropout,
        time_stride=time_stride,
        shapes_after_upsize=shapes_before_pooling
    ) for inst in range(nb_instruments)]

    decoded_inst = [decoders[inst](rnn_output) for inst in
                    range(nb_instruments)]  # List(nb_instruments)[(batch, step_length, size, channels)]
    outputs = [mlayers.last.LastInstMono(softmax_axis=-2)(decoded_inst[inst]) for inst in
               range(nb_instruments)]  # List(nb_instruments)[(batch, nb_steps=1, step_size, size, channels=1)]

    model = KerasModel(inputs=inputs_midi, outputs=outputs)

    # ------------------ Losses -----------------
    # Define losses dict for outputs
    losses = {}
    for inst in range(nb_instruments):
        losses[f'Output_{inst}'] = mlosses.loss_function_mono
    # ------------------ Metrics -----------------

    # ------------------------------ Compile ------------------------------

    model.compile(loss=losses,
                  optimizer=optimizer,
                  metrics=[mmetrics.acc_mono])
    model.build([(None, *midi_shape) for inst in range(nb_instruments)])

    return model, losses, (lambda_loss_activation, lambda_loss_duration)
