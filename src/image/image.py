import numpy as np
from PIL import Image
from pathlib import Path


def save_img(array, path):
    """

    :param array: shape (nb_instruments, 128, nb_steps, 2)
    :return:
    """
    activations = array[:, :, :, 0]      # (nb_instruments, 128, nb_steps)
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
        #img.show()
        img.save(save_path)


def show_image(array):
    activations = array[:, :, 0]
    np.place(activations, 0.5 <= activations, 1)
    np.place(activations, activations < 0.5, 0)
    img = Image.fromarray(activations)
    img.show()


