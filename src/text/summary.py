from pathlib import Path


def summarize_compute_data(path, **d):
    path = Path(path)
    # Add title:
    text = '\t\t' + d['data_name'] + '\n\n'\
            'instruments :' + str(d['instruments']) + '\n'\
            'input_size :' + str(d['input_size']) + '\n'\
            'notes_range :' + str(d['notes_range']) + '\n'\
            '\n'\
            'nb instruments :' + str(d['nb_instruments']) + '\n'\
            'nb files : ' + str(d['nb_files']) + '\n'\

    with open(str(path / 'Summary.txt'), 'a') as f:
        f.write(text)


def summarize_train(path, **d):
    path = Path(path)
    text = '\t\t' + d['full_name'] + '\n\n'\
            'epochs :' + str(d['epochs']) + '\n'\
            'input_param :' + str(d['input_param']) + '\n'\
            'instruments :' + str(d['instruments']) + '\n'\
            'notes_range :' + str(d['notes_range']) + '\n'
    with open(str(path / 'Summary.txt'), 'a') as f:
        f.write(text)
