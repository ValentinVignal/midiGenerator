import pygame.midi
from argparse import ArgumentParser

from src import Midi
from src.Midi.Player import Controller, MidiPlayer
from src import GlobalVariables as g


def main(args):
    controller = Controller(instrument=args.inst, tempo=args.tempo, work_on=g.mg.work_on)
    controller.play()
    for i, arr in enumerate(controller.arrays):
        print(f'Step {i} --> {arr[:, 60:92]}')
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
    args = parser.parse_args()
    args =preprocess(args)

    main(args)
