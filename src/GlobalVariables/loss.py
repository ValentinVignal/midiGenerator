# ----- Loss -----
loss_name = 'basic'
lambdas_loss = '2,2'


def get_lambdas_loss(lambdas_loss):
    l = lambdas_loss.split(',')
    return float(l[0]), float(l[1])


lambda_loss_activation, lambda_loss_duration = get_lambdas_loss(lambdas_loss)
l_scale = 1e-0
l_rhythm = 1e-1
l_scale_cost = 1e-3
l_rhythm_cost = 1e-5
take_all_step_rhythm = False




