import argparse
import os
from pathlib import Path
import shutil
from termcolor import colored

import src.midi.open as midi_open
import src.midi.create as midi_create


def all_midi_files(path, small_data):
    """

    :param path: the root path
    :param small_data: if we want to keep only a small amount of data
    :return: An array of all the path of all the .mid files in the directory
    """
    nb_small_data = 10
    fichiers = []
    if small_data:
        j = 0
        for root, dirs, files in os.walk(path):
            if j == nb_small_data:
                break
            for i in files:
                if j == nb_small_data:
                    break
                if i.endswith('.mid'):
                    fichiers.append(os.path.join(root, i))
                    j += 1
    else:
        for root, dirs, files in os.walk(path):
            for i in files:
                if i.endswith('.mid'):
                    fichiers.append(os.path.join(root, i))

    return fichiers


def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_mini',
                        help='The name of the data')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')

    args = parser.parse_args()

    if args.pc:
        data_path = Path(os.path.join('../Dataset', args.data))
    else:
        data_path = Path(os.path.join('../../../../../../storage1/valentin', args.data))
    data_checked_path = Path(data_path.as_posix() + '_checked')
    if data_checked_path.exists():  # Delete the folder of the transformed data
        shutil.rmtree(data_checked_path.as_posix())
    shutil.copytree(src=data_path.as_posix(), dst=data_checked_path)

    # Instruments :
    instruments = ['Piano', 'Trombone']

    all_midi_paths = all_midi_files(data_checked_path.as_posix(), False)
    nb_files = len(all_midi_paths)
    for i in range(nb_files):
        midi_path = Path(all_midi_paths[i])
        checked_file_name = Path(midi_path.parent, midi_path.stem +  '_checked' + midi_path.suffix)
        print(colored("-- {0}/{1} ----- : ----- Checking {2} ----------".format(i+1, nb_files, midi_path), 'white', 'on_blue'))
        matrix_midi = midi_open.midi_to_matrix(midi_path.as_posix(), instruments, print_instruments=True)  # (nb_instruments, 128, nb_steps, 2)
        if matrix_midi is None: continue
        #matrix_midi = np.transpose(matrix_midi, , 3))
        output_notes = midi_create.matrix_to_midi(matrix_midi, instruments=instruments)
        midi_create.save_midi(output_notes, instruments, checked_file_name.as_posix())




if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
