import os


def get_data_folder_path(args):
    """

    :param args:
    :return:
    """

    if args.pc:
        return os.path.join('..', 'Dataset')
    else:
        return os.path.join('..', '..', '..', '..', '..', '..', 'storage1', 'valentin')


def get_data_path(name, pc, transposed, mono):
    """

    :param name:
    :param pc:
    :param transposed:
    :param mono:
    :return:
    """
    if pc:
        # args.data = 'lmd_matched_mini'
        data_path = os.path.join('../Dataset', name)
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', name)
    data_transformed_path = data_path + '_transformed'
    if transposed:
        data_transformed_path += 'Transposed'
    if mono:
        data_transformed_path += 'Mono'
    return data_transformed_path



