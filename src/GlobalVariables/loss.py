# ----- Loss -----
loss_name = 'basic'
lambdas_loss = '2,2'


def get_lambdas_loss(lambdas_loss):
    l = lambdas_loss.split(',')
    return float(l[0]), float(l[1])


lambda_loss_activation, lambda_loss_duration = get_lambdas_loss(lambdas_loss)
l_scale = 1e-1
l_rhythm = 1e-2
take_all_step_rhythm = False
l_semitone = 1e-2
l_tone = 1e-2
l_tritone = 1e-2
use_binary = False




