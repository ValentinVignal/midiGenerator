import tensorflow as tf

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

    midi_shape = (nb_steps, input_size, 2)  # (batch, nb_step, input_size, 2)
    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, input_size, 2)]

    # ---------- Separated network for instruments ----------
    first_layer = []
    for instrument in range(nb_instruments):
        x_a = Lambda(lambda x: x[:, :, :, 0])(inputs_midi[instrument])  # activation (batch, nb_steps, input_size)
        x_d = Lambda(lambda x: x[:, :, :, 1])(inputs_midi[instrument])  # duration (batch, nb_steps, input_size)

        # 1
        x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='elu')(inputs_midi[instrument])
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        # 2
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='elu')(x)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)



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

    x = layers.Dense(nb_instruments * input_size)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.4)(x)

    # ---------- Instruments separately ----------
    outputs = []        # (batch, nb_steps, nb_instruments, input_size)
    for instrument in range(nb_instruments):
        output = layers.Dense(2 * input_size, activation='tanh')(x)
        output = layers.Reshape((input_size, 2))(output)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs_midi, outputs=outputs)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model
