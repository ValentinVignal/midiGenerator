import tensorflow as tf

"""

Model for a pc with little cpu (testing)

"""


def create_model(input_param, model_param, nb_steps):
    """

    :param input_param:
    :param model_param:
    :param nb_steps:
    :return: the neural network
    """
    print('Definition of the graph ...')

    nb_instruments = input_param['nb_instruments']
    input_size = input_param['input_size']

    midi_shape = (nb_steps, input_size)  # (batch, nb_step, input_size)
    inputs_midi = []
    for instrument in range(nb_instruments):
        inputs_midi.append(tf.keras.Input(midi_shape))  # [(batch, nb_steps, input_size)]

    # First level of lstm separated
    first_layer = []
    for instrument in range(nb_instruments):
        x = tf.keras.layers.LSTM(256, return_sequences=True, unit_forget_bias=True)(
            inputs_midi[instrument])  # (batch, nb_steps, 512)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        first_layer.append(x)

    # Concatenation
    for instrument in range(nb_instruments):
        first_layer[instrument] = tf.keras.layers.Reshape((nb_steps, 1, 256))(
            first_layer[instrument])  # (batch, nb_steps, 1, 256)

    x = tf.keras.layers.concatenate(first_layer, axis=2)  # (batch, nb_steps, nb_instruments, 256)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(nb_instruments * input_size)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = []        # (batch, nb_steps, nb_instruments, input_size)
    for instrument in range(nb_instruments):
        output = tf.keras.layers.Dense(input_size, activation='softmax')(x)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs_midi, outputs=outputs)

    return model
