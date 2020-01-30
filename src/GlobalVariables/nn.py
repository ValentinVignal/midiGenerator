# ----- Optimizer -----
lr = 5e-2
dropout = 1e-2
decay = 1e-1
epochs_drop = 50
decay_drop = 0.1

# ----- Architecture -----
all_sequence = False
lstm_state = False
last_fc = False
sampling = True
kld = True
kld_annealing_start = 4e-1
kld_annealing_stop = 7.5e-0
kld_sum = True
