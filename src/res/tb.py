"""
File to compute the graph of the results from the train on the server
"""
import os
import matplotlib.pyplot as plt
import argparse
import sys

from epicpath import EPath

sys.path.append(os.path.abspath('.'))

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


def plot_several_tensorboard(data, label, run_names=None, run_labels=None, max_length=None, train=True,
                             validation=False):
    """

    :param run_labels:
    :param validation:
    :param train:
    :param max_length:
    :param run_names:
    :param data:
    :param label:
    :return:
    """
    run_names = list(data.keys()) if run_names is None else run_names
    if run_labels is None:
        run_labels = run_names
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure()
    plt.title(label)
    validation_linestyle = '--' if train else '-'
    for i, run_name in enumerate(run_names):
        train, validation = data[run_name]['train'][label], data[run_name]['validation'][label]
        if train:
            plt.plot(train[0][:max_length], train[1][:max_length], label=f'Train {run_labels[i]}', linestyle='-',
                     color=colors[i])
        if validation:
            plt.plot(validation[0][:max_length], validation[1][:max_length], label=f'Val {run_labels[i]}',
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


def get_input_tensorboard_command(names=None):
    """

    :param names: list of strings
    :return:
    """
    if names is None:
        names = os.listdir(EPath(*['..' for _ in range(3)], 'big_runs').str)
    root = EPath(*['..' for _ in range(3)], 'big_runs')
    inputs = [
        f'{name}:/{(root / name / "saved_models" / os.listdir(root / name / "saved_models")[0] / "tensorboard").as_posix()}'
        for name in names
    ]
    return ','.join(inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get the logdir for tensorboard',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    print(os.listdir('.'))
    parser.add_argument('-n', '--names', default=None, help='The considered names')
    args = parser.parse_args()
    if args.names is not None:
        args.names = args.names.split(',')
    res = get_input_tensorboard_command(args.names)
    print(res)





