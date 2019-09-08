import tensorflow as tf

import src.eval_string as es
import src.NN.losses as l
import src.global_variables as g

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda

"""

First personal model

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
    mmodel_options ={
        'dropout': g.dropout,
        'all_sequence': g.all_sequence,
        'lstm_state': g.lstm_state
    }
    mmodel_options.update(model_options)

    # --------- End model options ----------

    print('Definition of the graph ...')

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    env = {
        'nb_instruments': nb_instruments,
        'input_size': input_size,
        'nb_steps': nb_steps
    }

    midi_shape = (nb_steps, step_length, input_size, 2)  # (batch, step_length, nb_step, input_size, 2)
    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, input_size, 2)]

    # --------- Only activation ----------
    inputs_activation = []
    for instrument in range(nb_instruments):
        inputs_activation.append(Lambda(lambda xl: xl[:, :, :, 0])(
            inputs_midi[instrument]))  # activation (batch, nb_steps, step_length, input_size)
        inputs_activation[instrument] = layers.Reshape((nb_steps, step_length, input_size, 1))(
            inputs_activation[instrument])

    # ---------- All together ----------
    x = layers.concatenate(inputs_midi, axis=4)  # (batch, nb_steps, step_length input_size, nb_instruments)

    # ----- Fully connected layers -----
    shape_before_fc = x.shape
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3] * x.shape[4]))(
        x)  # (batch, nb_steps, length * size * input_size)

    x = layers.Dropout(0.5)(x)
    # fc 1
    x = layers.TimeDistributed(layers.Dense(64))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    # fc 2
    x = layers.TimeDistributed(layers.Dense(48))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    # fc 2
    x = layers.TimeDistributed(layers.Dense(32))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # ---------- LSTM -----------
    x = layers.LSTM(64,
                    return_sequences=False,
                    return_state=False,
                    unit_forget_bias=True,
                    dropout=0.3,
                    recurrent_dropout=0.3)(x)  # (batch, nb_steps, size)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # ----- Fully Connected ------
    # fc 3
    x = layers.Dense(32)(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    # fc 2
    x = layers.Dense(48)(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    # fc 1
    x = layers.Dense(64)(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # ---------- Instruments separately ----------
    outputs = []  # (batch, nb_steps, nb_instruments, input_size)
    for instrument in range(nb_instruments):
        o = x
        output_a = layers.Dense(step_length * input_size, activation='sigmoid')(o)  # (batch, input_size)
        output_a = layers.Reshape((step_length, input_size, 1))(output_a)  # (batch, input_size, 1)
        output_d = layers.Dense(step_length * input_size, activation='sigmoid')(o)  # (batch, input_size)
        output_d = layers.Reshape((step_length, input_size, 1))(output_d)  # (batch, input_size, 1)
        output = layers.concatenate([output_a, output_d], axis=3)  # (batch, input_size, 2)
        output = layers.Layer(name='Output_{0}'.format(instrument))(output)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs_midi, outputs=outputs)

    # ------------------ Losses -----------------
    lambda_activation = 20
    lambda_duration = 0

    # Define losses dict
    losses = {}
    for i in range(nb_instruments):
        losses['Output_{0}'.format(i)] = l.choose_loss(type_loss)(lambda_activation, lambda_duration)

    model.compile(loss=losses, optimizer=optimizer, metrics=[l.acc_act, l.mae_dur])

    return model, losses, (lambda_activation, lambda_duration)
