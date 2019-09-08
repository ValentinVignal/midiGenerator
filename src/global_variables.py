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

noise = 0.002
