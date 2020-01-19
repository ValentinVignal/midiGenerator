# ------------------------------------------------------------
# ---------- For Midi files ----------
# ------------------------------------------------------------
max_length_note_music21 = 4
step_per_beat = 4
max_length_note_array = max_length_note_music21 * step_per_beat


# ------------------------------------------------------------
# ---------- For the computed Midi files ----------
# ------------------------------------------------------------
nb_files_per_npy = 20

# ------------------------------------------------------------
# ---------- Model ID ----------
# ------------------------------------------------------------
work_on = 'measure'
split_model_id = ','


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


# ------------------------------------------------------------
# ---------- Neural Network ----------
# ------------------------------------------------------------
# ----- Optimizer -----
lr = 0.01
dropout = 0.2
decay = 2e-1
epochs_drop = 50
decay_drop = 0.1
# ----- Loss -----
loss_name = 'basic'
lambdas_loss = '2,2'


def get_lambdas_loss(lambdas_loss):
    l = lambdas_loss.split(',')
    return float(l[0]), float(l[1])


lambda_loss_activation, lambda_loss_duration = get_lambdas_loss(lambdas_loss)
l_scale = 1
l_rhythm = 1
l_scale_cost = 1
l_rhythm_cost = 1
take_all_step_rhythm = False
# ----- Architecture -----
all_sequence = False
lstm_state = False
last_fc = False
sampling = True
kld = True
kld_annealing_start = 0.4
kld_annealing_stop = 0.8
kld_sum = True

# ----- Training -----
epochs = 200
batch = 8
validation = 0.1
# --- Data ---
noise = 0






