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


