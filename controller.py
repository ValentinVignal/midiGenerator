import pygame.midi
from argparse import ArgumentParser
import os

from src import Midi
from src.Midi.Player import MidiPlayer, BandPlayer
from src.MidiGenerator import MidiGenerator

os.system('echo Start Controller')


def main(args):
    midi_generator = MidiGenerator()
    midi_generator.recreate_model(args.load, print_model=False)

    controller = BandPlayer(instrument=args.inst,
                            tempo=args.tempo,
                            model=midi_generator,
                            played_voice=args.played_voice,
                            include_output=not args.no_include_output,
                            instrument_mask=args.inst_mask,
                            max_plotted=args.nb_steps_shown
                            )
    controller.play()
    """
    for i, arr in enumerate(controller.arrays):
        print(f'Step {i} --> {arr[:, 60:92]}')
    """
    MidiPlayer.player.close()
    pygame.midi.quit()


def preprocess(args):
    if args.inst == 'None':
        args.inst = None
    else:
        try:
            args.inst = int(args.inst)
        except ValueError:
            args.inst = Midi.instruments.all_midi_instruments.index(args.inst)
    if args.nb_steps_shown == -1:
        args.nb_steps_shown = None
    args.inst_mask = eval(args.inst_mask)
    return args


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inst', type=str, default='None',
                        help='Number or name of the instrument')
    parser.add_argument('--tempo', type=int, default=120,
                        help='Tempo')
    parser.add_argument('-l', '--load', type=str, default='',
                        help='The name of the train model to load')
    parser.add_argument('--no-include-output', default=False, action='store_true',
                        help='If set to True, the the input of the model is only what it is played')
    parser.add_argument('--played-voice', type=int, default=0,
                        help='The number of the voice played')
    parser.add_argument('--inst-mask', default=str(None), type=str,
                        help='Mask to hide some data, as a list of 0 and 1, 0=hide, 1=keep')
    parser.add_argument('--nb-steps-shown', type=int, default=-1,
                        help='Number of steps shown in the image, if -1 -> nb_steps model')
    args = parser.parse_args()
    args = preprocess(args)

    main(args)
