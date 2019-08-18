import tensorflow as tf

import src.eval_string as es
import src.NN.losses as l

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
    separated = model_param['separated1']
    for instrument in range(nb_instruments):
        x_a = Lambda(lambda xl: xl[:, :, :, 0])(inputs_midi[instrument])  # activation (batch, nb_steps, input_size)
        x_d = Lambda(lambda xl: xl[:, :, :, 1])(inputs_midi[instrument])  # duration (batch, nb_steps, input_size)

        x = inputs_midi[instrument]

        for i in range(len(separated['filters'])):
            for j in range(len(separated['filters'][i])):
                x = layers.Conv2D(filters=es.eval_all(separated['filters'][i][j], env=env),
                                  kernel_size=(3, 3),
                                  padding='same')(x)
                x = layers.LeakyReLU()(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.1)(x)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        """
        # from tutorial :
        
        x = tf.keras.layers.LSTM(1024, return_sequences=True, unit_forget_bias=True)(
            inputs_midi[instrument])  # (batch, nb_steps, 512)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # compute importance for each step
        attention = tf.keras.layers.Dense(1, activation='tanh')(x)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(1024)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)

        multiplied = tf.keras.layers.Multiply()([x, attention])
        sent_representation = tf.keras.layers.Dense(512)(multiplied)

        x = tf.keras.layers.Dense(512)(sent_representation)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.22)(x)

        # compute importance for each step
        attention = tf.keras.layers.Dense(1, activation='tanh')(x)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(512)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)

        multiplied = tf.keras.layers.Multiply()([x, attention])
        sent_representation = tf.keras.layers.Dense(256)(multiplied)

        x = tf.keras.layers.Dense(256)(sent_representation)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.22)(x)
        """

        first_layer.append(x)

    # ---------- Concatenation ----------
    # for instrument in range(nb_instruments):
    #     first_layer[instrument] = tf.keras.layers.Reshape((nb_steps, 1, 256))(
    #         first_layer[instrument])  # (batch, nb_steps, 1, 128)

    x = layers.concatenate(first_layer, axis=3)  # (batch, nb_steps, input_size, 16 * nb_instruments)

    # ---------- All together ----------
    x = layers.Flatten()(x)

    common = model_param['common']
    for i in range(len(common['fc'])):
        x = layers.Dense(es.eval_all(common['fc'][i], env=env))(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.4)(x)

    # ---------- Instruments separately ----------
    separated2 = model_param['separated2']
    outputs = []        # (batch, nb_steps, nb_instruments, input_size)
    for instrument in range(nb_instruments):
        o = x
        for i in range(len(separated2['fc'])):
            o = layers.Dense(es.eval_all(separated2['fc'][i], env=env))(o)
            o = layers.LeakyReLU()(o)
            o = layers.BatchNormalization()(o)
            o = layers.Dropout(0.3)(o)
        output_a = layers.Dense(input_size, activation='sigmoid')(o)    # (batch, input_size)
        output_a = layers.Reshape((input_size, 1))(output_a)        # (batch, input_size, 1)
        output_d = layers.Dense(input_size, activation='sigmoid')(o)      # (batch, input_size)
        output_d = layers.LeakyReLU()(output_d)
        output_d = layers.Reshape((input_size, 1))(output_d)        # (batch, input_size, 1)
        output = layers.concatenate([output_a, output_d], axis=2)      # (batch, input_size, 2)
        output = layers.Layer(name='Output_{0}'.format(instrument))(output)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs_midi, outputs=outputs)

    # ------------------ Loss -----------------
    lambda_activation = 20
    lambda_duration = 1

    """
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
    """


    # Define losses dict
    losses = {}
    for i in range(nb_instruments):
        key = 'Output_{0}'.format(i)
        losses[key] = l.custom_loss(lambda_activation, lambda_duration)

    model.compile(loss=losses, optimizer=optimizer)

    return model, losses, (lambda_activation, lambda_duration)
