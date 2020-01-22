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


