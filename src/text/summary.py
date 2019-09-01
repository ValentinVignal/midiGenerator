from pathlib import Path
from termcolor import cprint


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


