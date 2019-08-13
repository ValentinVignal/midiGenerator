import tensorflow as tf

import src.eval_string as es

layers = tf.keras.layers
Lambda = tf.keras.layers.Lambda

"""

First personal model

"""


def create_model(input_param, model_param, nb_steps, optimizer):
    """

    :param input_param:
    :param nb_steps:
    :param model_param:
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

    # ---------- Separated network for instruments ----------
    first_layer = []
    for instrument in range(nb_instruments):
        x_a = Lambda(lambda xl: xl[:, :, :, 0])(inputs_midi[instrument])  # activation (batch, nb_steps, input_size)
        x_d = Lambda(lambda xl: xl[:, :, :, 1])(inputs_midi[instrument])  # duration (batch, nb_steps, input_size)

        x = layers.Reshape((nb_steps, input_size * 2))(inputs_midi[instrument])  # (batch, nb_steps, 2 * input_size)

        for s in model_param['LSTM_separated']:
            size = int(s * nb_steps)
            x = tf.keras.layers.LSTM(size, return_sequences=True, unit_forget_bias=True)(x_a)  # (batch, nb_steps, size)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)

        first_layer.append(x)
    """
    # ---------- Concatenation ----------
    # for instrument in range(nb_instruments):
    #     first_layer[instrument] = tf.keras.layers.Reshape((nb_steps, 1, 256))(
    #         first_layer[instrument])  # (batch, nb_steps, 1, 128)

    x = layers.concatenate(first_layer, axis=2)  # (batch, nb_steps, size * nb_instruments)

    # ---------- All together ----------
    for s in model_param['LSTM_common']:
        size = int(s * nb_steps * nb_instruments)
        x = layers.LSTM(size, return_sequences=True, unit_forget_bias=True)(x)  # (batch, nb_steps, size)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    for s in model_param['fc_common']:
        size = s * nb_instruments
        x = layers.Dense(size)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.4)(x)
    """

    # ---------- Instruments separately ----------
    outputs = []  # (batch, nb_steps, nb_instruments, input_size)
    for instrument in range(nb_instruments):
        o = first_layer[instrument]
        for s in model_param['fc_separated']:
            o = layers.Dense(s)(o)
            o = layers.LeakyReLU()(o)
            o = layers.BatchNormalization()(o)
            o = layers.Dropout(0.3)(o)
        output_a = layers.Dense(input_size, activation='sigmoid')(o)  # (batch, input_size)
        output_a = layers.Reshape((input_size, 1))(output_a)  # (batch, input_size, 1)
        output_d = layers.Dense(input_size, activation='sigmoid')(o)  # (batch, input_size)
        output_d = layers.LeakyReLU()(output_d)
        output_d = layers.Reshape((input_size, 1))(output_d)  # (batch, input_size, 1)
        output = layers.concatenate([output_a, output_d], axis=2)  # (batch, input_size, 2)
        output = layers.Layer(name='Output_{0}'.format(instrument))(output)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs_midi, outputs=outputs)

    # ------------------ Losses -----------------
    lambda_activation = 20
    lambda_duration = 0

    def custom_loss(lambda_a, lambda_d):

        def loss_function(y_true, y_pred):
            y_true_a = Lambda(lambda x: x[:, :, 0])(y_true)
            y_true_d = Lambda(lambda x: x[:, :, 1])(y_true)
            y_pred_a = Lambda(lambda x: x[:, :, 0])(y_pred)
            y_pred_d = Lambda(lambda x: x[:, :, 1])(y_pred)

            loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
            loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

            loss = lambda_a * loss_a + lambda_d * loss_d

            return loss

        return loss_function

    # Define losses dict
    losses = {}
    for i in range(nb_instruments):
        losses['Output_{0}'.format(i)] = custom_loss(lambda_activation, lambda_duration)

    model.compile(loss=losses, optimizer=optimizer)

    return model, custom_loss(lambda_activation, lambda_duration)
