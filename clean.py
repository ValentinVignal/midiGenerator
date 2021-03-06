"""
File to clean environment
"""
from epicpath import EPath
import shutil

from src import Args
from src.Args import Parser, ArgType
from src import GlobalVariables as g


def main(args):
    """

    :param args:
    :return:
    """
    if not args.midi:
        shutil.rmtree(path='generated_midis', ignore_errors=True)
    if not args.hp:
        shutil.rmtree(path='hp_search', ignore_errors=True)
    if not args.model:
        shutil.rmtree(path='saved_models', ignore_errors=True)
    if not args.tensorboard:
        shutil.rmtree(path='tensorboard', ignore_errors=True)
    if not args.temp:
        shutil.rmtree(path='temp', ignore_errors=True)
    if not args.data_temp:
        data_temp = EPath(g.path.get_data_folder_path(args)) / 'temp'
        shutil.rmtree(path=data_temp, ignore_errors=True)
    if not args.zip:
        zip_path = EPath('my_zip.zip')
        if zip_path.exists():
            zip_path.unlink()
    print('Done cleaning')


if __name__ == '__main__':
    parser = Parser(argtype=ArgType.Clean)
    args = parser.parse_args()

    args = Args.preprocess.clean(args)
    main(args)
