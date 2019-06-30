import tensorflow as tf

def my_model(input_param):
    """

    :param input_param:
    :return: the neural network
    """
    print('Definition of the graph ...')

    midi_shape = (input_param['nb_steps'], input_param['input_size'])

    input_midi = tf.keras.Input(midi_shape)

    x = tf.keras.layers.LSTM(512, return_sequences=True, unit_forget_bias=True)(input_midi)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    """
    x = tf.keras.layers.LSTM(1024, return_sequences=True, unit_forget_bias=True)(input_midi)
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

    x = tf.keras.layers.LSTM(512, return_sequences=True, unit_forget_bias=True)(x)
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

    x = tf.keras.layers.LSTM(128, unit_forget_bias=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.22)(x)
    """
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='softmax')(x)

    model = tf.keras.Model(input_midi, x)

    return model
