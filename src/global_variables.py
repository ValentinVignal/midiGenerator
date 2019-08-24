# ---------- For midi files ----------
max_length_note_music21 = 4
step_per_beat = 4
max_length_note_array = max_length_note_music21 * step_per_beat


# ---------- For the computed midi files ----------
nb_file_per_npy = 20

# ---------- Neural Network ----------
dropout = 0.4
type_loss = 'smooth_round'
all_sequence = False
