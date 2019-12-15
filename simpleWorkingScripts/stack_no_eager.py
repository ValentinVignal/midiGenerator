import tensorflow as tf
import numpy as np
import math
import os
import time
import sys

root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.extend([root_path])

from src.NN import layers as mlayers

EAGER = True
DECAY = False

EPOCHS = 5

if not EAGER:
    tf.compat.v1.disable_eager_execution()

log_dir = os.path.join('tensorboard', f'{time.time()}')
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


def decay_func(lr_init):
    def step_decay(epoch):
        lrate = lr_init * math.pow(0.1, math.floor(epoch / 10))
        return lrate

    return step_decay


decay = tf.keras.callbacks.LearningRateScheduler(decay_func(0.1))


class MySequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size):
        super(MySequence, self).__init__()
        self.batch_size = batch_size

    def __len__(self):
        return 200

    def __getitem__(self, item):
        x = np.expand_dims(np.arange(20), axis=1) + np.random.rand(self.batch_size, 20, 30)
        y = np.expand_dims(np.arange(20, 40), axis=1) + np.random.rand(self.batch_size, 20, 10)
        return x, y


my_sequence = MySequence(batch_size=4)


def build_model():
    inputs = tf.keras.Input(shape=(20, 30))
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20))(inputs)
    # x = tf.keras.layers.LSTM(20, return_sequences=True)(x)
    x = mlayers.rnn.LstmRNN(size_list=[20, 30], return_sequence=True)(x)
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = build_model()

optimizer = tf.keras.optimizers.Adam(lr=0.01, decay=0.5)
model.compile(optimizer='adam', loss='mae')

model.summary()


start_train = time.time()
callbacks = [tensorboard]
if DECAY:
    callbacks.append(decay)
history = model.fit_generator(generator=my_sequence, epochs=EPOCHS, callbacks=callbacks,  validation_data=my_sequence,
                              shuffle=True)
end = time.time()


min_train, sec_train = int((end - start_train) // 60), int((end - start_train) % 60)
print(f'Time to train: {min_train}min{sec_train}sec')



