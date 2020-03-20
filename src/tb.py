"""
TensorBoard file
"""
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def get_tensorboard_data(path, train=True, validation=True):
    """
    Get the scalar data from the tensorboard files
    :param path:
    :return: dict(
        train=dict(
            scale_name=(x_array, y_array)
        ),
        validation=dict(
            scale_name=(x_array, y_array)
        )
    )
    """
    path = Path(path)

    data = {}
    keys = []
    if train:
        keys.append('train')
    if validation:
        keys.append('validation')

    for k in keys:
        data[k] = {}
        event_acc = event_accumulator.EventAccumulator((path / k).as_posix())
        event_acc.Reload()
        for tag in sorted(event_acc.Tags()['scalars']):
            x, y = [], []
            for scale_event in event_acc.Scalars(tag):
                x.append(scale_event.step)
                y.append(scale_event.value)
            data[k][tag] = (np.asarray(x), np.asarray(y))
    return data


def save_tensorboard_plots(data, path, mono=False):
    """

    :param mono:
    :param data:
    :param path:
    :return:
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    nb_instrument = 0
    while f'epoch_Output_{nb_instrument}_loss' in data['train']:
        nb_instrument += 1

    # --------------------------------------------------
    #               Cross train validation
    # --------------------------------------------------

    # Accuracy and loss of instruments
    suffixes = ['loss', 'acc_bin', 'acc_cat'] if mono else ['loss', 'acc']
    for s in suffixes:
        for i in range(nb_instrument):
            name = f'Output_{i}_{s}'
            key = f'epoch_{name}'
            plt.figure()
            train, validation = data['train'][key], data['validation'][key]
            plt.plot(train[0], train[1], label='Train', color=colors[0], linestyle='-')
            plt.plot(validation[0], validation[1], label='Validation', color=colors[1], linestyle='-')
            plt.title(name)
            plt.xlabel('Epochs')
            plt.ylabel(f'{s} value')
            plt.legend()
            plt.grid()
            plt.savefig((path / name).with_suffix('.png'))
            plt.close()
    # Total loss and KLD
    loss_names = ['loss', 'kld']
    for name in loss_names:
        key = f'epoch_{name}'
        plt.figure()
        train, validation = data['train'][key], data['validation'][key]
        plt.plot(train[0], train[1], label='Train', color=colors[0], linestyle='-')
        plt.plot(validation[0], validation[1], label='Validation', color=colors[1], linestyle='-')
        plt.title(name)
        plt.xlabel('Epochs')
        plt.ylabel(f'{name} value')
        plt.legend()
        plt.grid()
        plt.savefig((path / name).with_suffix('.png'))
        plt.close()

    # --------------------------------------------------
    #               Cross instruments
    # --------------------------------------------------

    # Accuracy and loss
    suffixes = ['loss', 'acc_bin', 'acc_cat'] if mono else ['loss', 'acc']
    for s in suffixes:
        name_file = f'Outputs_{s}'
        plt.figure()
        for i in range(nb_instrument):
            name_inst = f'Output_{i}'
            key = f'epoch_{name_inst}_{s}'
            train, validation = data['train'][key], data['validation'][key]
            plt.plot(train[0], train[1], label=f'{name_inst} train', color=colors[i], linestyle='-')
            plt.plot(validation[0], validation[1], label=f'{name_inst} val', color=colors[i], linestyle='--')
        plt.title(name_file)
        plt.xlabel('Epochs')
        plt.ylabel(f'{s} value')
        plt.legend()
        plt.grid()
        plt.savefig((path / name_file).with_suffix('.png'))
        plt.close()

    # Loss and KLD
    loss_names = ['loss', 'kld']
    plt.figure()
    for i in range(len(loss_names)):
        name = loss_names[i]
        key = f'epoch_{name}'
        train, validation = data['train'][key], data['validation'][key]
        plt.plot(train[0], train[1], label=f'{name} train', color=colors[i], linestyle='-')
        plt.plot(validation[0], validation[1], label=f'{name} val', color=colors[i], linestyle='--')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel(f'Losses value')
    plt.legend()
    plt.grid()
    plt.savefig((path / 'Loss_KLD').with_suffix('.png'))
    plt.close()






