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


def save_pianoroll(array, path, seed_length, instruments, one_note=False):
    """

    :param one_note:
    :param array: shape (nb_instruments, 128, nb_steps, 2)
    :param path:
    :param seed_length:
    :param instruments:
    :return:
    """
    # Colors
    colors_rgb = return_colors(len(instruments))
    if one_note:
        activations = array[:, :-1]  # (nb_instruments, 88, nb_steps)
    else:
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


def see_compare_generation_step(inputs, outputs, truth):
    """

    :param inputs: (nb_instruments, batch=2, nb_steps, step_size, input_size, 2)
    :param outputs: (nb_instruments, batch=2, step_size, input_size, 2)
    :param truth: (nb_instruments, batch=1, step_size, input_size, 2)
    :return:
    """
    print('inputs:', inputs.shape, 'outputs:', outputs.shape)
    inputs_a = inputs[:, :, :, :, :, 0]  # (nb_instruments, batch, nb_steps, step_size, input_size)
    np.place(inputs_a, 0.5 <= inputs_a, 1)
    np.place(inputs_a, inputs_a < 0.5, 0)
    outputs_a = outputs[:, :, :, :, 0]  # (nb_instruments, batch, step_size, input_size)
    np.place(outputs_a, 0.5 <= outputs_a, 1)
    np.place(outputs_a, outputs_a < 0.5, 0)
    nb_instruments, batch, nb_steps, step_size, input_size = inputs_a.shape
    inputs_a = np.reshape(inputs_a, (
        nb_instruments, batch, nb_steps * step_size, input_size))  # (nb_instruments, batch, length, input_size)

    truth_a = truth[:, 0, :, :, 0]  # (nb_instruments, step_size, input_size)

    inputs_alone = inputs_a[:, 0]  # (nb_instruments, length, input_size)
    inputs_helped = inputs_a[:, 1]
    outputs_alone = outputs_a[:, 0]
    outputs_helped = outputs_a[:, 1]

    colors = return_colors(nb_instruments)

    final_inputs_alone = np.zeros((nb_steps * step_size, input_size, 3))
    final_outputs_alone = np.zeros((step_size, input_size, 3))
    final_inputs_helped = np.zeros((nb_steps * step_size, input_size, 3))
    final_outputs_helped = np.zeros((step_size, input_size, 3))
    final_truth = np.zeros((step_size, input_size, 3))

    print('final_inputs_helped', np.count_nonzero(final_outputs_alone))
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
                if truth_a[inst, i, j] == 1:
                    final_truth[i, j] = colors[inst]
    print(final_outputs_alone.shape)
    final_inputs_alone = (np.flip(np.transpose(final_inputs_alone, (1, 0, 2)), axis=0)).astype(np.int)
    final_outputs_alone = (np.flip(np.transpose(final_outputs_alone, (1, 0, 2)), axis=0)).astype(np.int)
    final_inputs_helped = (np.flip(np.transpose(final_inputs_helped, (1, 0, 2)), axis=0)).astype(np.int)
    final_outputs_helped = (np.flip(np.transpose(final_outputs_helped, (1, 0, 2)), axis=0)).astype(np.int)
    final_truth = (np.flip(np.transpose(final_truth, (1, 0, 2)), axis=0)).astype(np.int)
    print('final_inputs_helped with colors', np.count_nonzero(final_outputs_alone))

    acc_alone_inst = [1 - (np.count_nonzero(outputs_alone[inst] - truth_a[inst]) / truth_a[inst].size) for
                      inst in range(nb_instruments)]
    acc_alone = sum(acc_alone_inst) / len(acc_alone_inst)
    acc_helped_inst = [1 - (np.count_nonzero(outputs_helped[inst] - truth_a[inst]) / truth_a[inst].size) for
                      inst in range(nb_instruments)]
    acc_helped = sum(acc_helped_inst) / len(acc_helped_inst)

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].imshow(final_inputs_alone)
    axs[0, 0].set_title('Inputs Alone')
    axs[0, 1].imshow(final_outputs_alone)
    axs[0, 1].set_title('Outputs Alone' + '\n' + 'Accuracy alone {0}, {1}'.format(acc_alone_inst, acc_alone))
    axs[1, 0].imshow(final_inputs_helped)
    axs[1, 0].set_title('Inputs Helped')
    axs[1, 1].imshow(final_outputs_helped)
    axs[1, 1].set_title('Outputs Helped' + '\n' + 'Accuracy helped {0}, {1}'.format(acc_helped_inst, acc_helped))
    axs[2, 0].imshow(final_inputs_helped)
    axs[2, 0].set_title('Inputs Truth')
    axs[2, 1].imshow(final_truth)
    axs[2, 1].set_title('Outputs Truth')
    plt.show()


def see_compare_on_batch(x, yt, yp):
    """

    :param x: (nb_instruments, batch, nb_steps, step_size, inputs_size, 2)
    :param yt: (nb_instruments, batch, step_size, input_size, 2)
    :param yp: (nb_instruments, batch, step_size, input_size, 2)
    :return:
    """
    # ----- Keep only activation -----
    x = np.array(x)[:, :, :, :, :, 0]
    yt = np.array(yt)[:, :, :, :, 0]
    np.place(yt, 0.5 <= yt, 1)
    np.place(yt, yt < 0.5, 0)
    yp = np.array(yp)[:, :, :, :, 0]
    np.place(yp, 0.5 <= yp, 1)
    np.place(yp, yp < 0.5, 0)

    nb_instruments, batch, nb_steps, step_size, input_size = x.shape
    x = np.reshape(x, (nb_instruments, batch, nb_steps * step_size, input_size))

    colors = return_colors(nb_instruments)

    x_final = np.zeros((batch, nb_steps * step_size, input_size, 3))
    yt_final = np.zeros((batch, step_size, input_size, 3))
    yp_final = np.zeros((batch, step_size, input_size, 3))
    for b in range(batch):
        for inst in range(nb_instruments):
            for i in range(nb_steps * step_size):
                for j in range(input_size):
                    if x[inst, b, i, j] == 1:
                        x_final[b, i, j] = colors[inst]
            for i in range(step_size):
                for j in range(input_size):
                    if yt[inst, b, i, j] == 1:
                        yt_final[b, i, j] = colors[inst]
                    if yp[inst, b, i, j] == 1:
                        yp_final[b, i, j] = colors[inst]
    x_final = (np.flip(np.transpose(x_final, (0, 2, 1, 3)), axis=1)).astype(np.int)
    yt_final = (np.flip(np.transpose(yt_final, (0, 2, 1, 3)), axis=1)).astype(np.int)
    yp_final = (np.flip(np.transpose(yp_final, (0, 2, 1, 3)), axis=1)).astype(np.int)

    fig, axs = plt.subplots(batch, 3)
    accs = np.zeros((batch, nb_instruments))
    for b in range(batch):
        accs[b] = np.array([1 - (np.count_nonzero(yt[inst, b] - yp[inst, b]) / yp[inst, b].size) for inst in range(nb_instruments)])
        acc = np.sum(accs[b]) / nb_instruments
        axs[b, 0].imshow(x_final[b])
        axs[b, 0].set_title('Input')
        axs[b, 1].imshow(yt_final[b])
        axs[b, 1].set_title('Truth')
        axs[b, 2].imshow(yp_final[b])
        axs[b, 2].set_title('Predicted\n{0}, {1}'.format(accs[b], acc))
    accs = np.sum(accs, axis=0) / len(accs)
    acc = np.sum(accs) / len(accs)
    plt.title('{0}, {1}'.format(accs, acc))
    plt.show()








