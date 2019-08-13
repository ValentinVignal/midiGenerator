import numpy as np
from PIL import Image
from pathlib import Path
from colour import Color


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
    colors = [Color(pick_for=instrument) for instrument in instruments]
    colors_rgb = list(map(lambda color: [int(255 * c) for c in list(color.get_rgb())], colors))

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
