from pathlib import Path

def summarise_compute_data(path, **dict):
    path = Path(path)
    text = ''
    # Add title:
    text += '\t\t' + dict['data_name'] + '\n\n'\
            'instruments :' + str(dict['instruments']) + '\n'\
            'input_size :' + str(dict['input_size']) + '\n'\
            'notes_range :' + str(dict['notes_range']) + '\n'\
            '\n'\
            'nb instruments :' + str(dict['nb_instruments']) + '\n'\
            'nb files : ' + str(dict['nb_files']) + '\n'\

    with open(str(path / 'Summary.txt'), 'a') as f:
        f.write(text)

