# ----- Optimizer -----
lr = 1e-3
opt_name = 'adam'
# dropout = 1e-2
dropout_d = 1e-2        # For dense layers
dropout_c = 1e-3        # For convolutional layers
dropout_r = 1e-3        # For RNN layers
decay = 1e-1
epochs_drop = 50
decay_drop = 0.1

# ----- Architecture -----
all_sequence = False
lstm_state = False
last_fc = False
sampling = True
kld = True
kld_annealing_start = 1e-2
kld_annealing_stop = 4e-1
kld_sum = True
sah = False
rpoe = True
prior_expert = True
