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

    controller = BandPlayer(instrument=args.inst, tempo=args.tempo, model=midi_generator)
    controller.play()
    """
    for i, arr in enumerate(controller.arrays):
        print(f'Step {i} --> {arr[:, 60:92]}')
    """
    MidiPlayer.player.close()
    pygame.midi.quit()


def preprocess(args):
    try:
        args.inst = int(args.inst)
    except ValueError:
        args.inst = Midi.instruments.all_midi_instruments.index(args.inst)
    return args


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inst', type=str, default='0',
                        help='Number or name of the instrument')
    parser.add_argument('--tempo', type=int, default=120,
                        help='Tempo')
    parser.add_argument('--voice', type=int, default=0,
                        help='The voice the player wants to play')
    parser.add_argument('-l', '--load', type=str, default='',
                      help='The name of the train model to load')
    args = parser.parse_args()
    args =preprocess(args)

    main(args)
