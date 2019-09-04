# ---------- For midi files ----------
max_length_note_music21 = 4
step_per_beat = 4
max_length_note_array = max_length_note_music21 * step_per_beat


# ---------- For the computed midi files ----------
nb_file_per_npy = 20

# ---------- Neural Network ----------
lr = 0.01
dropout = 0.2
epochs_drop = 50
decay_drop = 0.5
type_loss = 'no_round'
all_sequence = False
lstm_state = False
work_on = 'beat'
