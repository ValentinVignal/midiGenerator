import tensorflow as tf
import numpy as np
K = tf.keras.backend

Lambda = tf.keras.layers.Lambda

# ---------- Real output ----------


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


def custom_loss_round(lambda_a, lambda_d):
    def loss_function(y_true, y_pred):
        y_true_a = Lambda(lambda x: x[:, :, 0])(y_true)
        y_true_d = Lambda(lambda x: x[:, :, 1])(y_true)
        y_pred_a = Lambda(lambda x: x[:, :, 0])(y_pred)
        y_pred_a_rounded = tf.round(y_pred_a)
        y_pred_d = Lambda(lambda x: x[:, :, 1])(y_pred)

        loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
        loss_a_rounded = tf.keras.losses.mean_squared_error(y_true_a, y_pred_a_rounded)
        loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

        loss = lambda_a * (loss_a + 5 * loss_a_rounded) + lambda_d * loss_d

        return loss

    return loss_function


def custom_loss_smoothround(lambda_a, lambda_d):
    def loss_function(y_true, y_pred):
        y_true_a = Lambda(lambda x: x[:, :, 0])(y_true)
        y_true_d = Lambda(lambda x: x[:, :, 1])(y_true)
        y_pred_a = Lambda(lambda x: x[:, :, 0])(y_pred)
        y_pred_d = Lambda(lambda x: x[:, :, 1])(y_pred)
        # Calcul of "rounded"
        a = 50
        y_pred_a_rounded = 1 / (1 + tf.math.exp(-a * (y_pred_a - 0.5)))

        loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
        loss_a_rounded = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a_rounded)
        loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

        loss = lambda_a * (loss_a + loss_a_rounded) + lambda_d * loss_d

        return loss

    return loss_function


def compare_losses():
    n = 20
    yt = np.random.uniform(size=(n, 2))       # Only activations
    yp = np.random.randint(2, size=(n, 2))       # Only activations
    for i in range(len(yt)):
        yt_ = yt[np.newaxis, np.newaxis, i]
        yp_ = yp[np.newaxis, np.newaxis, i]
        F = custom_loss(1, 1)(K.variable(yt_), K.variable(yp_))
        F_rounded = custom_loss_round(1, 1)(K.variable(yt_), K.variable(yp_))
        F_smooth = custom_loss_smoothround(1, 1)(K.variable(yt_), K.variable(yp_))
        print('Truth :{0}, Pred :{1} -- loss {2}, round {3}, smoothround {4}'.format(yt[i], yp[i], K.eval(F), K.eval(F_rounded), K.eval(F_smooth)))


def choose_loss(type_loss):
    if type_loss == 'no_round':
        return custom_loss
    elif type_loss == 'rounded':
        return custom_loss_round
    elif type_loss == 'smooth_round':
        return custom_loss_smoothround
    else:
        raise Exception('type_loss "{0}" not known'.format(type_loss))


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
    y_true_a = Lambda(lambda x: x[:, :, 0])(y_true)
    y_pred_a = Lambda(lambda x: x[:, :, 0])(y_pred)

    acc = tf.keras.metrics.binary_accuracy(y_true_a, y_pred_a)

    return acc


def mae_dur(y_true, y_pred):
    y_true_d = Lambda(lambda x: x[:, :, 1])(y_true)
    y_pred_d = Lambda(lambda x: x[:, :, 1])(y_pred)

    mae = tf.keras.metrics.mean_squared_error(y_true_d, y_pred_d)

    return mae
