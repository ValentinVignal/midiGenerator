import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice

from sklearn import datasets

"""
K = tf.keras.backend        instead of import tf.python.keras.backend
layers = tf.keras.layers        instead of import tf.python.keras.layers

Because on the server, it raise an error saying there is no module tf.python (I don't know why)
"""
K = tf.keras.backend
layers = tf.keras.layers

np.random.seed(0)
# ----- Creation of the data -----
n_samples = 8
noisy_moons = datasets.make_circles(n_samples=n_samples, noise=0.05)
nm_inputs = np.repeat(noisy_moons[0], 24, axis=0)
nm_truths = np.repeat(noisy_moons[1], 24, axis=0)
n_clusters = 2

"""
colors = np.array(list(
    islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']),
           int(max(nm_truths) + 1))))
plt.scatter(nm_inputs[:, 0], nm_inputs[:, 1], color=colors[nm_truths])
plt.show()
"""

# ----- Definition of the model -----

inputs = layers.Input((2,))  # (2, )
# ---
x = layers.Dense(10)(inputs)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
# ---
x = layers.Reshape((10, 1))(x)
# ---
x = layers.Conv1D(filters=4, kernel_size=3, strides=1, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
# ---
x = layers.Conv1D(filters=4, kernel_size=3, strides=1, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
# ---
x = layers.Flatten()(x)
# ---
x = layers.Dense(20)(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
# ---
x = layers.Dense(20)(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
# ---
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4)
model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
model.fit(nm_inputs, nm_truths, verbose=1, shuffle=True, epochs=100, batch_size=8)
evaluation = model.evaluate(nm_inputs, nm_truths, verbose=1)
print('evaluation', evaluation)
