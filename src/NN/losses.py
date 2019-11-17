import tensorflow as tf
import numpy as np
K = tf.keras.backend

Lambda = tf.keras.layers.Lambda

# ---------- Real output ----------


def custom_loss(lambda_a, lambda_d):
    def loss_function(y_true, y_pred):
        """

        :param y_true: (batch, lenght, input_size, 2)
        :param y_pred: (batch, lenght, input_size, 2)
        :return:
        """
        y_true_a = Lambda(lambda x: x[:, :, :, 0])(y_true)
        y_true_d = Lambda(lambda x: x[:, :, :, 1])(y_true)
        y_pred_a = Lambda(lambda x: x[:, :, :, 0])(y_pred)
        y_pred_d = Lambda(lambda x: x[:, :, :, 1])(y_pred)

        loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
        loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

        loss = lambda_a * loss_a + lambda_d * loss_d

        return tf.reduce_mean(loss, axis=None)

    return loss_function


def custom_loss_round(lambda_a, lambda_d):
    def loss_function(y_true, y_pred):
        y_true_a = Lambda(lambda x: x[:, :, :, 0])(y_true)
        y_true_d = Lambda(lambda x: x[:, :, :, 1])(y_true)
        y_pred_a = Lambda(lambda x: x[:, :, :, 0])(y_pred)
        y_pred_a_rounded = tf.round(y_pred_a)
        y_pred_d = Lambda(lambda x: x[:, :, :, 1])(y_pred)

        loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
        loss_a_rounded = tf.keras.losses.mean_squared_error(y_true_a, y_pred_a_rounded)
        loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

        loss = lambda_a * (loss_a + 5 * loss_a_rounded) + lambda_d * loss_d

        return tf.reduce_mean(loss, axis=None)

    return loss_function


def custom_loss_smoothround(lambda_a, lambda_d):
    def loss_function(y_true, y_pred):
        y_true_a = Lambda(lambda x: x[:, :, :, 0])(y_true)
        y_true_d = Lambda(lambda x: x[:, :, :, 1])(y_true)
        y_pred_a = Lambda(lambda x: x[:, :, :, 0])(y_pred)
        y_pred_d = Lambda(lambda x: x[:, :, :, 1])(y_pred)
        # Calcul of "rounded"
        a = 50
        y_pred_a_rounded = 1 / (1 + tf.math.exp(-a * (y_pred_a - 0.5)))

        loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
        loss_a_rounded = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a_rounded)
        loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

        loss = lambda_a * (loss_a + loss_a_rounded) + lambda_d * loss_d

        return tf.reduce_mean(loss, axis=None)

    return loss_function


def custom_loss_linearround(lambda_a, lambda_d):

    def loss_function(y_true, y_pred):
        y_true_a = Lambda(lambda x: x[:, :, :, 0])(y_true)
        y_true_d = Lambda(lambda x: x[:, :, :, 1])(y_true)
        y_pred_a = Lambda(lambda x: x[:, :, :, 0])(y_pred)
        y_pred_d = Lambda(lambda x: x[:, :, :, 1])(y_pred)
        # Calcul of "rounded"
        a = 50
        y_pred_a_rounded = (0.8 / (1 + tf.math.exp(-a * (y_pred_a - 0.5)))) + 0.2 * y_pred_a

        loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
        loss_a_rounded = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a_rounded)
        loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

        loss = lambda_a * (loss_a + loss_a_rounded) + lambda_d * loss_d

        return tf.reduce_mean(loss, axis=None)

    return loss_function


def custom_loss_duration(lambda_a, lambda_d):

    def filter_duration(inputs):
        p_d = inputs[0]
        t_a = inputs[1]
        return tf.math.multiply(p_d, t_a)

    def loss_dur(t_d, p_d):
        diff = (t_d - p_d)
        pow = tf.math.multiply(diff, diff)
        return tf.reduce_sum(pow, axis=2)

    def loss_function(y_true, y_pred):
        y_true_a = Lambda(lambda x: x[:, :, :, 0])(y_true)
        y_true_d = Lambda(lambda x: x[:, :, :, 1])(y_true)
        y_pred_a = Lambda(lambda x: x[:, :, :, 0])(y_pred)
        y_pred_d = Lambda(lambda x: x[:, :, :, 1])(y_pred)

        loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)

        y_pred_d = Lambda(filter_duration)([y_pred_d, y_true_a])
        # loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)
        loss_d = loss_dur(y_true_d, y_pred_d)

        loss = lambda_a * loss_a + lambda_d * loss_d

        return tf.reduce_mean(loss, axis=None)

    return loss_function


def compare_losses_random(n=20):
    yt = np.random.randint(2, size=(n, 2))       # Only activations
    yp = np.random.randint(2, size=(n, 2))       # Only activations
    for i in range(len(yt)):
        yt_ = yt[np.newaxis, np.newaxis, np.newaxis, i]
        yp_ = yp[np.newaxis, np.newaxis, np.newaxis, i]
        F = custom_loss(1, 1)(K.variable(yt_), K.variable(yp_))
        F_rounded = custom_loss_round(1, 1)(K.variable(yt_), K.variable(yp_))
        F_smooth = custom_loss_smoothround(1, 1)(K.variable(yt_), K.variable(yp_))
        print('Truth :{0}, Pred :{1} -- loss {2}, round {3}, smoothround {4}'.format(yt[i], yp[i], K.eval(F), K.eval(F_rounded), K.eval(F_smooth)))


def compare_losses_auto(step=0.1):

    yp_a = np.arange(0, 1+step, step)  # Only activations
    yp = np.zeros((yp_a.shape[0], 2))
    yp[:, 0] = yp_a
    yt = np.array([[0, 0], [1, 0]])
    for i in range(len(yp)):
        yt_ = yt[np.newaxis, np.newaxis, np.newaxis, 0]
        yp_ = yp[np.newaxis, np.newaxis, np.newaxis, i]
        F = custom_loss(1, 1)(K.variable(yt_), K.variable(yp_))
        F_rounded = custom_loss_round(1, 1)(K.variable(yt_), K.variable(yp_))
        F_smooth = custom_loss_smoothround(1, 1)(K.variable(yt_), K.variable(yp_))
        print('Truth :{0}, Pred :{1} -- loss {2}, round {3}, smoothround {4}'.format(yt[0], yp[i], K.eval(F),
                                                                                     K.eval(F_rounded),
                                                                                     K.eval(F_smooth)))
    for i in range(len(yp)):
        yt_ = yt[np.newaxis, np.newaxis, np.newaxis, 1]
        yp_ = yp[np.newaxis, np.newaxis, np.newaxis, i]
        F = custom_loss(1, 1)(K.variable(yt_), K.variable(yp_))
        F_rounded = custom_loss_round(1, 1)(K.variable(yt_), K.variable(yp_))
        F_smooth = custom_loss_smoothround(1, 1)(K.variable(yt_), K.variable(yp_))
        print('Truth :{0}, Pred :{1} -- loss {2}, round {3}, smoothround {4}'.format(yt[1], yp[i], K.eval(F),
                                                                                     K.eval(F_rounded),
                                                                                     K.eval(F_smooth)))


def choose_loss(type_loss):
    if type_loss == 'no_round':
        return custom_loss
    elif type_loss == 'rounded':
        return custom_loss_round
    elif type_loss == 'smooth_round':
        return custom_loss_smoothround
    elif type_loss == 'linear_round':
        return custom_loss_linearround
    elif type_loss == 'dur':
        return custom_loss_duration
    else:
        raise Exception('type_loss "{0}" not known'.format(type_loss))


def loss_function_mono(y_true, y_pred):
    """

    :param y_true: (batch, nb_steps=1, step_size, input_size, channels=1)
    :param y_pred: (batch, nb_steps=1, step_size, input_size, channels=1)
    :return:
    """
    y_true_a = Lambda(lambda x: tf.gather(x, axis=4, indices=0))(y_true)
    y_pred_a = Lambda(lambda x: tf.gather(x, axis=4, indices=0))(y_pred)
    y_true_a_no_nan = tf.where(tf.math.is_nan(y_true_a), tf.zeros_like(y_true_a), y_true_a)
    y_pred_a_no_nan = tf.where(tf.math.is_nan(y_pred_a), tf.zeros_like(y_pred_a), y_pred_a)

    loss = tf.keras.losses.categorical_crossentropy(y_true_a_no_nan, y_pred_a_no_nan)
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    return loss

# ---------- LSTM ----------


def custom_losslstm():
    def loss_functionlstm(y_true, y_pred):
        loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        return loss
    return loss_functionlstm


def choose_losslstm(type_loss_lstm):
    return custom_losslstm


# --------- Metrics ----------


def acc_act(y_true, y_pred):
    y_true_a = Lambda(lambda x: x[:, :, :, 0])(y_true)
    y_pred_a = Lambda(lambda x: x[:, :, :, 0])(y_pred)

    acc = tf.keras.metrics.binary_accuracy(y_true_a, y_pred_a)

    return tf.reduce_mean(acc, axis=None)


def mae_dur(y_true, y_pred):
    y_true_d = Lambda(lambda x: x[:, :, :, 1])(y_true)
    y_pred_d = Lambda(lambda x: x[:, :, :, 1])(y_pred)

    mae = tf.keras.metrics.mean_squared_error(y_true_d, y_pred_d)

    return tf.reduce_mean(mae, axis=None)


def acc_mono(y_true, y_pred):
    y_true_a = Lambda(lambda x: x[:, :, :, :, 0])(y_true)
    y_pred_a = Lambda(lambda x: x[:, :, :, :, 0])(y_pred)

    acc = tf.keras.metrics.categorical_accuracy(y_true_a, y_pred_a)

    return tf.reduce_mean(acc, axis=None)

