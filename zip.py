"""
To zip everything
"""
from zipfile import ZipFile
import os

from src import Args
from src.Args import Parser, ArgType


def get_all_file_paths(directory):
    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

            # returning all file paths
    return file_paths


def main(args):
    """

    :param args:
    :return:
    """
    roots = []
    if args.all:
        if not args.no_midi:
            roots.append('generated_midis')
        if not args.no_hp:
            roots.append('search_hp')
        if not args.no_model:
            roots.append('saved_models')
    else:
        if args.midi:
            roots.append('generated_midis')
        if args.hp:
            roots.append('hp_search')
        if args.model:
            roots.append('saved_models')

    to_zip = []
    for root in roots:
        to_zip.extend(get_all_file_paths(root))

    with ZipFile('my_zip.zip', 'w') as zip:
        for file in to_zip:
            zip.write(file)

    print('Done zipping in my_zip.zip')


if __name__ == '__main__':
    parser = Parser(argtype=ArgType.Zip)
    args = parser.parse_args()

    args = Args.preprocess.zip(args)
    main(args)