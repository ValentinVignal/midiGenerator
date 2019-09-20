from pathlib import Path
from termcolor import cprint
import matplotlib.pyplot as plt


def summarize_compute_data(path, **d):
    path = Path(path)
    # Add title:
    text = '\t\t' + d['data_name'] + '\n\n'

    text += 'instruments :' + str(d['instruments']) + '\n'
    text += 'input_size :' + str(d['input_size']) + '\n'
    text += 'notes_range :' + str(d['notes_range']) + '\n'
    text += '\n'
    text += 'nb instruments :' + str(d['nb_instruments']) + '\n'
    text += 'nb files : ' + str(d['nb_files']) + '\n'

    file_path = path / 'Summary.txt'
    if file_path.exists():
        file_path.unlink()
    with open(str(file_path), 'a') as f:
        f.write(text)


def summarize_train(path, **d):
    path = Path(path)
    text = '\t\t' + d['full_name'] + '\n\n'

    text += 'epochs :' + str(d['epochs']) + '\n'
    text += 'input_param :' + str(d['input_param']) + '\n'
    text += 'instruments :' + str(d['instruments']) + '\n'
    text += 'notes_range :' + str(d['notes_range']) + '\n'
    text += 'work_on :' + d['work_on'] + '\n'

    file_path = path / 'Summary.txt'
    if file_path.exists():
        file_path.unlink()
    with open(str(file_path), 'a') as f:
        f.write(text)


def summarize_generation(path, **d):
    path = Path(path)
    text = '\t\t' + d['full_name'] + '\n\n'

    text += 'epochs :' + str(d['epochs']) + '\n'
    text += 'input_param :' + str(d['input_param']) + '\n'
    text += 'instruments :' + str(d['instruments']) + '\n'
    text += 'notes_range :' + str(d['notes_range']) + '\n'

    file_path = path / 'Summary.txt'
    if file_path.exists():
        file_path.unlink()
    with open(str(file_path), 'a') as f:
        f.write(text)


def summarize_loss_history(path, logs, paths, hparams, best_index=None):
    path = Path('tests_hp')
    path.mkdir(parents=True, exist_ok=True)
    if best_index is not None:
        text = '\t\t---------- Best model ----------\n\n' \
               'Best index : {0} --> Loss : {4}\n' \
               'Logs : {1}\n' \
               'Hyper Parameters : {2}\n' \
               'Save Path : {3}\n\n\n'.format(best_index, logs[best_index], hparams[best_index], paths[best_index],
                                              logs[best_index]['loss'])
    else:
        text = ''

    text += '\t\t---------- All informations ---------- \n\n'

    for j in range(len(logs)):
        text += 'Index : {0} --> Loss : {1}\n' \
                'Logs : {2}\n' \
                'Hyper Parameters : {3}\n' \
                'Save Path : {4}\n\n'.format(j, logs[j]['loss'], logs[j], hparams[j], paths[j])

    with open(str(path), 'a') as f:
        f.write(text)
    cprint('Summary saved at {0}'.format(path), 'green')


def update_summary_loss_history(path, log, path_model, hparam, j):
    text = 'Index : {0} --> Loss : {1}\n' \
            'Logs : {2}\n' \
            'Hyper Parameters : {3}\n' \
            'Save Path : {4}\n\n'.format(j, log['loss'], log, hparam, path_model)

    with open(str(path), 'a') as f:
        f.write(text)
    cprint('Summary updated at {0}'.format(path), 'green')


def update_best_summary_loss_history(path, log, path_model, hparam, best_index):
    text = '\t\t---------- Best model ----------\n\n' \
           'Best index : {0} --> Loss : {4}\n' \
           'Logs : {1}\n' \
           'Hyper Parameters : {2}\n' \
           'Save Path : {3}\n\n\n'.format(best_index, log, hparam, path_model,
                                          log['loss'])

    text += '\t\t---------- All informations ---------- \n\n'
    with open(path, 'r') as f:
        ftext = text + f.read()
    with open(path, 'w') as f2:
        f2.write(ftext)
    cprint('Summary updated with the best model at {0}'.format(path), 'green')


def save_train_history(train_history, nb_instruments, pathlib):
    """

    :param train_history:
    :param nb_instruments:
    :param pathlib:
    :return:
    """

    print('history')
    print(train_history)
    loss = train_history['loss']
    val_loss = train_history['val_loss']
    losses = [train_history['Output_{0}_loss'.format(i)] for i in range(nb_instruments)]
    val_losses = [train_history['val_Output_{0}_loss'.format(i)] for i in range(nb_instruments)]
    accs_act = [train_history['Output_{0}_acc_act'.format(i)] for i in range(nb_instruments)]
    val_accs_act = [train_history['val_Output_{0}_acc_act'.format(i)] for i in range(nb_instruments)]
    mae_dur = [train_history['Output_{0}_mae_dur'.format(i)] for i in range(nb_instruments)]
    val_mae_dur = [train_history['val_Output_{0}_mae_dur'.format(i)] for i in range(nb_instruments)]

    epochs = range(1, len(loss) + 1)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ----- Save losses -----
    plt.figure()
    plt.plot(epochs, loss, label='Loss', color=colors[0], linestyle='-')
    plt.plot(epochs, val_loss, label='val_Loss', color=colors[0], linestyle='--')
    for i in range(nb_instruments):
        plt.plot(epochs, losses[i], label='Output_{0}_loss'.format(i), color=colors[i+1], linestyle='-')
        plt.plot(epochs, val_losses[i], label='val_Output_{0}_loss'.format(i), color=colors[i+1], linestyle='--')

    plt.title(f'Variation of the Loss through the epochs\nLoss : {loss[-1]}\nval_Loss : {val_loss[-1]}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    plt.legend()
    plt.grid()
    plt.savefig((pathlib / 'Losses.png').as_posix())

    # ----- Save accuracies -----
    plt.figure()
    for i in range(nb_instruments):
        plt.plot(epochs, accs_act[i], label='Output_{0}_acc_act'.format(i), color=colors[i+1], linestyle='-')
        plt.plot(epochs, val_accs_act[i], label='val_Output_{0}_acc_act'.format(i), color=colors[i+1], linestyle='--')

    plt.title('Variation of the accuracy through the epochs\n')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy value')
    plt.legend()
    plt.grid()
    plt.savefig((pathlib / 'Accuracies_activation.png').as_posix())

    # ----- Save durations -----
    plt.figure()
    for i in range(nb_instruments):
        plt.plot(epochs, mae_dur[i], label='Output_{0}_acc_act'.format(i), color=colors[i+1], linestyle='-')
        plt.plot(epochs, val_mae_dur[i], label='val_Output_{0}_acc_act'.format(i), color=colors[i+1], linestyle='--')

    plt.title('Variation of the accuracy mae through the epochs\n')
    plt.xlabel('Epochs')
    plt.ylabel('Duration mae value')
    plt.legend()
    plt.grid()
    plt.savefig((pathlib / 'Duration_mae.png').as_posix())

    # ----- Text -----
    text = 'Loss : {0}'.format(loss[-1])
    text += '\n\n'
    text += 'By instrument :\n'
    for i in range(nb_instruments):
        text += '\tInstrument {0}\n'.format(i)
        text += '\t\t Loss : {0}\t--\t{1}\n'.format(losses[i][-1], losses[i])
        text += '\t\t Validation Loss : {0}\t--\t{1}\n'.format(val_losses[i][-1], val_losses[i])
        text += '\t\t Accuracy activation : {0}\t--\t{1}\n'.format(accs_act[i][-1], accs_act[i])
        text += '\t\t Validation Accuracy activation : {0}\t--\t{1}\n'.format(val_accs_act[i][-1], val_accs_act[i])
        text += '\t\t Duration mae : {0}\t--\t{1}\n'.format(mae_dur[i][-1], mae_dur[i])
        text += '\t\t Validation Duration mae : {0}\t--\t{1}\n'.format(val_mae_dur[i][-1], val_mae_dur[i])
    with open((pathlib / 'Summary.txt').as_posix(), 'w') as f:
        f.write(text)

