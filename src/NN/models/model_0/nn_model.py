import tensorflow as tf

"""

Model gave by the tutorial

"""


def create_model(input_param, model_param, nb_steps, optimizer):
    """

    :param input_param:
    :param model_param:
    :param nb_steps:
    :param optimizer:
    :return: the neural network
    """
    print('Definition of the graph ...')

    # ---------------------------------------
    # ----------- Neural network ------------
    # ---------------------------------------

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    midi_shape = (nb_steps, input_size)  # (batch, nb_step, input_size)
    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, input_size)]

    # First level of lstm separated
    first_layer = []
    for instrument in range(nb_instruments):
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

        first_layer.append(x)

    # Concatenation
    for instrument in range(nb_instruments):
        first_layer[instrument] = tf.keras.layers.Reshape((nb_steps, 1, 256))(
            first_layer[instrument])  # (batch, nb_steps, 1, 128)

    x = tf.keras.layers.concatenate(first_layer, axis=2)  # (batch, nb_steps, nb_instruments, 128)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(nb_instruments * input_size)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = []        # (batch, nb_steps, nb_instruments, input_size)
    for instrument in range(nb_instruments):
        output = tf.keras.layers.Dense(input_size, activation='softmax')(x)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs_midi, outputs=outputs)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model
