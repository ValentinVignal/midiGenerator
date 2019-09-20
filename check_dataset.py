import argparse
import os
from pathlib import Path
import shutil
from termcolor import colored

import src.midi.open as midi_open
import src.midi.create as midi_create
import src.midi.common as midi_common
import src.midi.instruments as midi_inst
import src.image.pianoroll as p


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
    parser.add_argument('--bach', action='store_true', default=False,
                        help='To compute the bach data')

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

    # Stat
    max_notes_range, min_notes_range = 0, int(s[1]) - int(s[0])
    nb_correct_files = 0
    nb_measures = 0

    # Instruments :
    args.instruments = list(map(lambda instrument: ' '.join(instrument.split('_')),
                                args.instruments.split(',')))
    if args.bach:
        args.instruments = midi_inst.bach_instruments
    print('\t', colored('Check_dataset with instruments : ', 'cyan', 'on_white') +
          colored('{0}'.format(args.instruments), 'magenta', 'on_white'), '\n')

    all_midi_paths = midi_open.all_midi_files(data_checked_path.as_posix(), False)
    nb_files = len(all_midi_paths)
    print('note_range:', colored(args.notes_range, 'magenta'))
    for i in range(nb_files):
        midi_path = Path(all_midi_paths[i])
        checked_file_name = Path(midi_path.parent, midi_path.stem + '_checked' + midi_path.suffix)
        checked_file_name_image = Path(midi_path.parent, midi_path.stem + '_checked.jpg')
        print(colored("-- {0}/{1} ----- : ----- Checking {2} ----------".format(i + 1, nb_files, midi_path), 'white',
                      'on_blue'))
        if args.bach:
            matrix_midi = midi_open.midi_to_matrix_bach(filename=midi_path.as_posix(),
                                                        print_instruments=True,
                                                        notes_range=args.notes_range
                                                        )
        else:
            matrix_midi = midi_open.midi_to_matrix(filename=midi_path.as_posix(),
                                                   instruments=args.instruments,
                                                   print_instruments=True,
                                                   notes_range=args.notes_range
                                                   )  # (nb_args.instruments, 128, nb_steps, 2)
        if matrix_midi is None:
            continue

        # Update stats
        matrix_bound_min, matrix_bound_max = midi_common.range_notes_in_matrix(matrix_midi)
        min_notes_range = min(min_notes_range, matrix_bound_min)
        max_notes_range = max(max_notes_range, matrix_bound_max)
        nb_correct_files += 1
        nb_measures += midi_common.nb_measures(matrix_midi)

        # matrix_midi = np.transpose(matrix_midi, , 3))
        output_notes = midi_create.matrix_to_midi(matrix_midi, instruments=args.instruments,
                                                  notes_range=args.notes_range)
        midi_create.save_midi(output_notes, args.instruments, checked_file_name.as_posix())
        p.save_pianoroll(matrix_midi, path=checked_file_name_image.as_posix(), seed_length=0,
                         instruments=args.instruments)
    min_notes_range += args.notes_range[0]
    max_notes_range += args.notes_range[0]
    print('----------', colored('Checking is done', 'white', 'on_green'), '----------')
    print('Number of correct files :', colored(nb_correct_files, 'magenta'), '-- nb measures :',
          colored(nb_measures, 'magenta'))
    print('Range of notes :', colored(min_notes_range, 'magenta'), ':', colored(max_notes_range, 'magenta'))


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
