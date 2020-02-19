import os
from pathlib import Path
import shutil
from termcolor import colored
import sys

sys.path.append(os.path.abspath('.'))

from src import Midi
from src import Images
from src import Args
from src.Args import ArgType, Parser

os.system('echo Start check dataset')


def main(args):
    """
        Entry point
    """

    if args.pc:
        data_path = Path(os.path.join('../Dataset', args.data))
    else:
        data_path = Path(os.path.join('../../../../../../storage1/valentin', args.data))
    data_checked_path = Path(data_path.as_posix() + '_checked')
    if data_checked_path.exists():  # Delete the folder of the transformed data
        shutil.rmtree(data_checked_path.as_posix())
    shutil.copytree(src=data_path.as_posix(), dst=data_checked_path)

    # Stat
    max_notes_range, min_notes_range = 0, int(args.notes_range[1]) - int(args.notes_range[0])
    nb_correct_files = 0
    nb_measures = 0

    print('\t', colored('Check_dataset with instruments : ', 'cyan', 'on_white') +
          colored('{0}'.format(args.instruments), 'magenta', 'on_white'), '\n')

    all_midi_paths = Midi.open.all_midi_files(data_checked_path.as_posix(), False)
    nb_files = len(all_midi_paths)
    print('note_range:', colored(args.notes_range, 'magenta'))
    for i in range(nb_files):
        midi_path = Path(all_midi_paths[i])
        checked_file_name = Path(midi_path.parent, midi_path.stem + '_checked' + midi_path.suffix)
        checked_file_name_image = Path(midi_path.parent, midi_path.stem + '_checked.jpg')
        print(colored("-- {0}/{1} ----- : ----- Checking {2} ----------".format(i + 1, nb_files, midi_path), 'white',
                      'on_blue'))
        if args.bach:
            matrix_midi = Midi.open.midi_to_matrix_bach(filename=midi_path.as_posix(),
                                                        print_instruments=True,
                                                        notes_range=args.notes_range
                                                        )
        else:
            matrix_midi = Midi.open.midi_to_matrix(filename=midi_path.as_posix(),
                                                   instruments=args.instruments,
                                                   print_instruments=True,
                                                   notes_range=args.notes_range,

                                                   )  # (nb_args.instruments, 128, nb_steps, 2)
        if matrix_midi is None:
            continue
        if args.mono:
            matrix_midi = Midi.open.to_mono_matrix(matrix_midi)

        # Update stats
        matrix_bound_min, matrix_bound_max = Midi.common.range_notes_in_matrix(matrix_midi)
        min_notes_range = min(min_notes_range, matrix_bound_min)
        max_notes_range = max(max_notes_range, matrix_bound_max)
        nb_correct_files += 1
        nb_measures += Midi.common.nb_measures(matrix_midi)

        # matrix_midi = np.transpose(matrix_midi, , 3))
        output_notes = Midi.create.matrix_to_midi(matrix_midi, instruments=args.instruments,
                                                  notes_range=args.notes_range, mono=args.mono)
        Midi.create.save_midi(output_notes, args.instruments, checked_file_name.as_posix())
        Images.pianoroll.save_array_as_pianoroll(
            array=matrix_midi,
            folder_path=midi_path.parent,
            name=midi_path.stem + '_checked',
            seed_length=0,
            mono=args.mono or args.bach
        )
    min_notes_range += args.notes_range[0]
    max_notes_range += args.notes_range[0]
    print('----------', colored('Checking is done', 'white', 'on_green'), '----------')
    print('Number of correct files :', colored(nb_correct_files, 'magenta'), '-- nb measures :',
          colored(nb_measures, 'magenta'))
    print('Range of notes :', colored(min_notes_range, 'magenta'), ':', colored(max_notes_range, 'magenta'))


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    parser = Parser(argtype=ArgType.CheckData)
    args = parser.parse_args()

    args = Args.preprocess.check_dataset(args)
    main(args)
