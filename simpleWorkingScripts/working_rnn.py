from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import math
import dill
import time
import argparse
import sys


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    sys.path.extend([root_path])
    from src.NN import layers as mlayers

    parser = argparse.ArgumentParser(description='Program to test eager execution',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eager', action='store_true', default=False,
                        help='Use Eager exectution')
    args = parser.parse_args()

if not args.eager:
    tf.compat.v1.disable_eager_execution()
start = time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


def decay_func(lr_init):
    def step_decay(epoch):
        lrate = lr_init * math.pow(0.1, math.floor(epoch / 10))
        return lrate

    return dill.loads(dill.dumps(step_decay))


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
    x = tf.keras.layers.TimeDistributed(mlayers.dense.DenseCoder(
        size_list=['10', '15', '20']
    ))(inputs)
    # x = tf.keras.layers.LSTM(20, return_sequences=True)(x)
    x = mlayers.rnn.LstmRNN(
        size_list=[30, 20, 10],
        dropout=0.1,
        return_sequence=True
    )(x)
    outputs = tf.keras.layers.TimeDistributed(mlayers.dense.DenseCoder(
        size_list=['20', '15', '10']
    ))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = build_model()
model.summary()


def loss(labels, logits):
    return tf.keras.losses.mae(labels, logits)


model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 5
decay = tf.keras.callbacks.LearningRateScheduler(decay_func(0.1))

start_train = time.time()
history = model.fit_generator(generator=my_sequence, epochs=EPOCHS, callbacks=[checkpoint_callback, decay])
end = time.time()


min_all, sec_all = int((end - start) // 60), int((end - start) % 60)
min_train, sec_train = int((end - start_train) // 60), int((end - start_train) % 60)
print(f'Time: over all: {min_all}min{sec_all}sec - train: {min_train}min{sec_train}sec')



