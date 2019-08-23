import argparse
import os
from pathlib import Path
import shutil
from termcolor import colored

import src.midi.open as midi_open
import src.midi.create as midi_create
import src.image.pianoroll as p


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
    parser.add_argument('data', type=str, default='lmd_matched_mini',
                        help='The name of the data')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    parser.add_argument('--instruments', type=str, default='Piano,Tuba',
                        help='The instruments considered (for space in name, put _ instead : Acoustic_Bass)')
    parser.add_argument('--images', action='store_true', default=False,
                        help='To also create the pianoroll')
    parser.add_argument('--notes-range', type=str, default='0:88',
                        help='The length of the data')

    args = parser.parse_args()

    if args.pc:
        data_path = Path(os.path.join('../Dataset', args.data))
    else:
        data_path = Path(os.path.join('../../../../../../storage1/valentin', args.data))
    data_checked_path = Path(data_path.as_posix() + '_checked')
    if data_checked_path.exists():  # Delete the folder of the transformed data
        shutil.rmtree(data_checked_path.as_posix())
    shutil.copytree(src=data_path.as_posix(), dst=data_checked_path)
    s = args.notes_range.split(':')
    args.notes_range = (int(s[0]), int(s[1]))

    # Instruments :
    instruments = ['Piano', 'Tuba', 'Flute', 'Violin']
    args.instruments = list(map(lambda instrument: ' '.join(instrument.split('_')),
                                args.instruments.split(',')))
    print('\t', colored('Check_dataset with instruments : ', 'cyan', 'on_white') +
          colored('{0}'.format(args.instruments), 'magenta', 'on_white'), '\n')

    all_midi_paths = all_midi_files(data_checked_path.as_posix(), False)
    nb_files = len(all_midi_paths)
    for i in range(nb_files):
        midi_path = Path(all_midi_paths[i])
        checked_file_name = Path(midi_path.parent, midi_path.stem + '_checked' + midi_path.suffix)
        checked_file_name_image = Path(midi_path.parent, midi_path.stem + '_checked.jpg')
        print(colored("-- {0}/{1} ----- : ----- Checking {2} ----------".format(i + 1, nb_files, midi_path), 'white',
                      'on_blue'))
        print('note_range:', args.notes_range)
        matrix_midi = midi_open.midi_to_matrix(midi_path.as_posix(), args.instruments, print_instruments=True,
                                               notes_range=args.notes_range)  # (nb_args.instruments, 128, nb_steps, 2)
        if matrix_midi is None:
            continue
        # matrix_midi = np.transpose(matrix_midi, , 3))
        output_notes = midi_create.matrix_to_midi(matrix_midi, instruments=args.instruments,
                                                  notes_range=args.notes_range)
        midi_create.save_midi(output_notes, args.instruments, checked_file_name.as_posix())
        p.save_pianoroll(matrix_midi, path=checked_file_name_image.as_posix(), seed_length=0,
                         instruments=args.instruments)


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
