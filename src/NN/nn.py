import importlib.util
import os


def create_nn_model(model_id, input_param):
    """

    :param model_id:
    :param input_param:
    :return: the neural network
    """

    model_name, model_param = model_id.split(';')

    path = os.path.join('src',
                        'NN',
                        'models',
                        'model_{0}'.format(model_name),
                        'nn_model.py')
    spec = importlib.util.spec_from_file_location('nn_model', path)
    nn_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nn_model)

    # TODO: load the parameters if model_param != None

    return nn_model.create_model(
        input_param=input_param,
        model_param=model_param)




