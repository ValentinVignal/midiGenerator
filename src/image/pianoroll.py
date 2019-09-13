import numpy as np
from PIL import Image
from pathlib import Path
from colour import Color
import random
import matplotlib.pyplot as plt


def save_img(array, path):
    """

    :param array: shape (nb_instruments, 128, nb_steps, 2)
    :return:
    """
    activations = array[:, :, :, 0]  # (nb_instruments, 128, nb_steps)
    np.place(activations, 0.5 <= activations, 1)
    np.place(activations, activations < 0.5, 0)
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    for i in range(len(activations)):
        save_path = (path / 'inst({0}).jpg'.format(i)).as_posix()
        a = np.array(255 * activations[i], dtype=int)
        values = []
        for j in range(len(activations[i])):
            for k in range(len(activations[i][j])):
                if a[j][k] not in values:
                    values.append(a[j][k])
        img = Image.fromarray(
            (255 * np.flip(activations[i], axis=0)).astype(np.uint8),
            mode='L')
        # img.show()
        img.save(save_path)


def show_image(array):
    activations = array[:, :, 0]
    np.place(activations, 0.5 <= activations, 1)
    np.place(activations, activations < 0.5, 0)
    img = Image.fromarray(activations)
    img.show()


def see_MySequence(x, y):
    """
    To see the output of MySequence class
    can use : see_MySequence(*mySequence[i])
    :param x:
    :param y:
    :return:
    """
    x, y = np.array(x)[0, 0, :, :, 0], np.array([y])[0, 0, 0, np.newaxis, :, 0]
    all = np.zeros((x.shape[0] + 1, x.shape[1], 3))
    all[:-1, :, 0] = x
    all[-1, :, 1] = y
    all = (255 * np.transpose(all, (1, 0, 2))).astype(np.uint8)
    img = Image.fromarray(all, mode='RGB')
    img.show()


def save_pianoroll(array, path, seed_length, instruments):
    """

    :param array: shape (nb_instruments, 128, nb_steps, 2)
    :param path:
    :param seed_length:
    :param instruments:
    :return:
    """
    # Colors
    colors = [Color('#' + ''.join([random.choice('0123456789abcdef') for j in range(6)])) for i in instruments]
    colors_rgb = list(map(lambda color: [int(255 * c) for c in list(color.get_rgb())], colors))
    for i in range(len(colors_rgb)):  # Make a light color
        m = min(colors_rgb[i])
        M = max(colors_rgb[i])
        if M <= 100:  # If the color is too dark
            for j in range(3):
                if colors_rgb[i][j] == M:
                    colors_rgb[i][j] = min(50 + 2 * colors_rgb[i][j], 255)
                elif colors_rgb[i][j] == m:
                    colors_rgb[i][j] = 10 + int(1.3 * colors_rgb[i][j])
                else:
                    colors_rgb[i][j] = 25 + int(1.7 * colors_rgb[i][j])

    activations = array[:, :, :, 0]  # (nb_instruments, 88, nb_steps)
    nb_instruments, input_size, nb_steps = activations.shape
    np.place(activations, 0.5 <= activations, 1)
    np.place(activations, activations < 0.5, 0)
    all = np.zeros((input_size, nb_steps, 3))  # RGB
    all[:, :seed_length] = 25  # So seed is visible (grey)
    for inst in range(nb_instruments):
        for i in range(input_size):
            for j in range(nb_steps):
                if activations[inst, i, j] == 1:
                    all[i, j] = colors_rgb[inst]
    img = Image.fromarray(
        np.flip(all, axis=0).astype(np.uint8),
        mode='RGB'
    )
    img.save(path)


def return_colors(nb_instruments):
    colors = [Color('#' + ''.join([random.choice('0123456789abcdef') for j in range(6)])) for i in
              range(nb_instruments)]
    colors_rgb = list(map(lambda color: [int(255 * c) for c in list(color.get_rgb())], colors))
    for i in range(len(colors_rgb)):  # Make a light color
        m = min(colors_rgb[i])
        M = max(colors_rgb[i])
        if M <= 100:  # If the color is too dark
            for j in range(3):
                if colors_rgb[i][j] == M:
                    colors_rgb[i][j] = min(50 + 3 * colors_rgb[i][j], 255)
                elif colors_rgb[i][j] == m:
                    colors_rgb[i][j] = min(10 + int(1.5 * colors_rgb[i][j]), 255)
                else:
                    colors_rgb[i][j] = min(25 + 2 * colors_rgb[i][j], 255)
    return colors_rgb


def see_compare_generation_step(inputs, outputs):
    """

    :param inputs: (nb_instruments, batch=2, nb_steps, step_size, input_size, 2)
    :param outputs: (nb_instruments, batch=2, step_size, input_size, 2)
    :return:
    """
    print('inputs:', inputs.shape, 'outputs:', outputs.shape)
    inputs_a = inputs[:, :, :, :, :, 0]  # (nb_instruments, batch, nb_steps, step_size, input_size)
    np.place(inputs_a, 0.5 <= inputs_a, 1)
    np.place(inputs_a, inputs_a > 0.5, 0)
    outputs_a = outputs[:, :, :, :, 0]  # (nb_instruments, batch, step_size, input_size)
    np.place(outputs_a, 0.5 <= outputs_a, 1)
    np.place(outputs_a, outputs_a > 0.5, 0)
    nb_instruments, batch, nb_steps, step_size, input_size = inputs_a.shape
    inputs_a = np.reshape(inputs_a, (
    nb_instruments, batch, nb_steps * step_size, input_size))  # (nb_instruments, batch, length, input_size)

    inputs_alone = inputs_a[:, 0]       # (nb_instruments, length, input_size)
    inputs_helped = inputs_a[:, 1]
    outputs_alone = outputs_a[:, 0]
    outputs_helped = outputs_a[:, 1]

    colors = return_colors(nb_instruments)

    final_inputs_alone = np.zeros((nb_steps * step_size, input_size, 3))
    final_outputs_alone = np.zeros((step_size, input_size, 3))
    final_inputs_helped = np.zeros((nb_steps * step_size, input_size, 3))
    final_outputs_helped = np.zeros((step_size, input_size, 3))

    for inst in range(nb_instruments):
        for i in range(nb_steps * step_size):
            for j in range(input_size):
                if inputs_alone[inst, i, j] == 1:
                    final_inputs_alone[i, j] = colors[inst]
                if inputs_helped[inst, i, j] == 1:
                    final_inputs_helped[i, j] = colors[inst]
        for i in range(step_size):
            for j in range(input_size):
                if outputs_alone[inst, i, j] == 1:
                    final_outputs_alone[i, j] = colors[inst]
                if outputs_helped[inst, i, j] == 1:
                    final_outputs_helped[i, j] = colors[inst]
    print(final_outputs_alone.shape)
    final_inputs_alone = (np.flip(np.transpose(final_inputs_alone, (1, 0, 2)), axis=0)).astype(np.int)
    final_outputs_alone = (np.flip(np.transpose(final_outputs_alone, (1, 0, 2)), axis=0)).astype(np.int)
    final_inputs_helped = (np.flip(np.transpose(final_inputs_helped, (1, 0, 2)), axis=0)).astype(np.int)
    final_outputs_helped = (np.flip(np.transpose(final_outputs_helped, (1, 0, 2)), axis=0)).astype(np.int)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(final_inputs_alone)
    axs[0, 0].set_title('Inputs Alone')
    axs[0, 1].imshow(final_outputs_alone)
    axs[0, 1].set_title('Outputs Alone')
    axs[1, 0].imshow(final_inputs_helped)
    axs[1, 0].set_title('Inputs Helped')
    axs[1, 1].imshow(final_outputs_helped)
    axs[1, 1].set_title('Outputs Helped')
    plt.show()




