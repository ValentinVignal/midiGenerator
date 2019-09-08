# ---------- For midi files ----------
max_length_note_music21 = 4
step_per_beat = 4
max_length_note_array = max_length_note_music21 * step_per_beat


# ---------- For the computed midi files ----------
nb_file_per_npy = 20

# ---------- Neural Network ----------
lr = 0.005
dropout = 0.2
epochs_drop = 50
decay_drop = 0.5
type_loss = 'linear_round'
all_sequence = False
lstm_state = False
work_on = 'beat'


def work_on2nb(wo):
    if wo == 'note':
        return 1
    elif wo == 'beat':
        return step_per_beat
    elif wo == 'measure':
        return 4 * step_per_beat
    else:
        raise Exception('Unknow work_on type :{0}'.format(wo))


def work_on2letter(wo):
    if wo == 'note':
        return 'n'
    elif wo == 'beat':
        return 'b'
    elif wo == 'measure':
        return 'm'
    elif wo is None:
        return ''
    else:
        raise Exception('Unknow work_on type :{0}'.format(wo))


def letter2work_on(letter):
    if letter == 'n':
        return 'note'
    elif letter == 'b':
        return 'beat'
    elif letter == 'm':
        return 'measure'
    elif letter == '':
        return None
    else:
        raise Exception('Unknow work_on letter :{0}'.format(letter))


noise = 0.002
