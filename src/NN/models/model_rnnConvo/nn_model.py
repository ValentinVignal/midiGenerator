import tensorflow as tf

import src.eval_string as es
import src.NN.losses as l
import src.global_variables as g

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda

"""

LSTM with encoder convolutional layers

"""


def create_model(input_param, model_param, nb_steps, optimizer, dropout=g.dropout, type_loss=g.type_loss):
    """

    :param input_param:
    :param nb_steps:
    :param model_param:
    :param optimizer:
    :param dropout: value of the dropout
    :param type_loss:
    :return: the neural network
    """

    print('Definition of the graph ...')

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    env = {
        'nb_instruments': nb_instruments,
        'input_size': input_size,
        'nb_steps': nb_steps
    }

    midi_shape = (nb_steps, input_size, 2)  # (batch, nb_step, input_size, 2)
    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, input_size, 2)]

    # --------- Only activation ----------
    inputs_activation = []
    for instrument in range(nb_instruments):
        inputs_activation.append(Lambda(lambda xl: xl[:, :, :, 0])(
            inputs_midi[instrument]))  # activation (batch, nb_steps, input_size)
        inputs_activation[instrument] = layers.Reshape((nb_steps, input_size, 1))(inputs_activation[instrument])

    # ---------- All together ----------
    x = layers.concatenate(inputs_midi, axis=3)  # (batch, nb_steps, input_size, nb_instruments)

    convo = model_param['convo']
    for i in convo:
        for s in i:
            size = s
            x = layers.Conv2D(filters=size, kernel_size=(1, 3), padding='same')(x)
            x = layers.LeakyReLU()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout / 2)(x)
        x = layers.MaxPool2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)  # (batch, size * filters)
    # ---------- LSTM -----------
    lstm = model_param['LSTM']
    for index, s in enumerate(lstm):
        size = int(s * nb_steps)
        x = layers.LSTM(size,
                        return_sequences=(index + 1) != len(lstm),
                        unit_forget_bias=True)(x)  # (batch, nb_steps, size)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

    # x : (batch, size)

    for s in model_param['fc_common']:
        size = es.eval_all(s, env)
        x = layers.Dense(size)(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

    # ---------- Instruments separately ----------
    outputs = []  # (batch, nb_steps, nb_instruments, input_size)
    for instrument in range(nb_instruments):
        o = x
        for s in model_param['fc_separated']:
            o = layers.Dense(s)(o)
            o = layers.LeakyReLU()(o)
            o = layers.BatchNormalization()(o)
            o = layers.Dropout(dropout)(o)
        output_a = layers.Dense(input_size, activation='sigmoid')(o)  # (batch, input_size)
        output_a = layers.Reshape((input_size, 1))(output_a)  # (batch, input_size, 1)
        output_d = layers.Dense(input_size, activation='sigmoid')(o)  # (batch, input_size)
        output_d = layers.Reshape((input_size, 1))(output_d)  # (batch, input_size, 1)
        output = layers.concatenate([output_a, output_d], axis=2)  # (batch, input_size, 2)
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

    return model, losses, (lambda_activation, lambda_duration),
