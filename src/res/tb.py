"""
File to compute the graph of the results from the train on the server
"""
import os
import matplotlib.pyplot as plt

from epicpath import EPath

from src import tb as tb


def name_to_foler(name):
    """

    :param name: name of the folder of the big run
    :return:
    """
    # Root Path
    path = EPath(*['..' for _ in range(3)], 'big_runs')  # Path where all the folders are
    # name
    return path / name


def get_all_data(paths):
    """

    :param paths: list of path of the different folder of saved models (EpicPath)
    :return:
    """
    data = {}
    for path in paths:
        name = path.name
        # Get the tensorboard folder
        tensorboard_folder = path / 'saved_models'
        tensorboard_folder = tensorboard_folder / os.listdir(tensorboard_folder)[0] / 'tensorboard'
        d = tb.get_tensorboard_data(tensorboard_folder)
        data[name] = d
    return data


def plot_several_tensorboard(data, label, run_labels=None, max_length=None, train=True, validation=False):
    """

    :param validation:
    :param train:
    :param max_length:
    :param run_labels:
    :param data:
    :param label:
    :return:
    """
    run_labels = list(data.keys()) if run_labels is None else run_labels
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure()
    plt.title(label)
    validation_linestyle = '--' if train else '-'
    for i, run_label in enumerate(run_labels):
        train, validation = data[run_label]['train'][label], data[run_label]['validation'][label]
        if train:
            plt.plot(train[0][:max_length], train[1][:max_length], label=f'Train {run_label}', linestyle='-',
                     color=colors[i])
        if validation:
            plt.plot(validation[0][:max_length], validation[1][:max_length], label=f'Val {run_label}',
                     linestyle=validation_linestyle, color=colors[i])
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()


def plot_from_names(names, label, *args, **kwargs):
    """

    :param label:
    :param names: list of strings
    :return:
    """
    paths = [name_to_foler(name) for name in names]
    data = get_all_data(paths)
    plot_several_tensorboard(*args, data=data, label=label, **kwargs)


