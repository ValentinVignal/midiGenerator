from pynput import keyboard
import pygame.midi
import time
from string import ascii_lowercase
from argparse import ArgumentParser

key_to_note = {
    'q': 72,  # C5
    '2': 73,
    'w': 74,  # D5
    '3': 75,
    'e': 76,  # E5
    'r': 77,  # F5
    '5': 78,
    't': 79,  # G5
    '6': 80,
    'y': 81,  # A5
    '7': 82,
    'u': 83,  # B5
    'i': 84,  # C6
    '9': 85,
    'o': 86,  # D6
    '0': 87,
    'p': 88,  # E6
    '[': 89,  # F6
    '=': 90,
    ']': 91,  # G6
    # ---
    'z': 60,  # C4,
    's': 61,
    'x': 62,  # D4
    'd': 63,
    'c': 64,  # E4
    'v': 65,  # F4
    'g': 66,
    'b': 67,  # G4
    'h': 68,
    'n': 69,  # A4
    'j': 70,
    'm': 71,  # B5
    ',': 72,  # C5
    'l': 73,
    '.': 74,  # D5
    ';': 75,
    '/': 76,  # E5
}

midi_instruments = [
    # Piano
    'Acoustic-Piano',
    'BrtAcou-Piano',
    'ElecGrand-Piano',
    'Honky-Tonk-Piano',
    'Elec.Piano-1',
    'Elec.Piano-2',
    'Harsichord',
    'Clavichord',
    # Chromatic Percussion
    'Celesta',
    'Glockenspiel',
    'Music-Box',
    'Vibraphone',
    'Marimba',
    'Xylophone',
    'Tubular-Bells',
    'Dulcimer',
    # Organ
    'Drawbar Organ',
    'Perc.Organ',
    'Rock-Organ',
    'Church-Organ',
    'Reed-Organ',
    'Accordian',
    'Harmonica',
    'Tango-Accordian',
    # Guitar
    'Acoustic-Guitar',
    'SteelAcous.Guitar',
    'El.Jazz-Guitar',
    'Electric-Guitar',
    'El.Muted-Guitar',
    'Overdriven-Guitar',
    'Distortion-Guitar',
    'Guitar-Harmonic',
    # Bass
    'Acoustic-Bass',
    'El.Bass-Finger',
    'El.Bass-Pick',
    'Fretless-Bass',
    'Slap Bass 1',
    'Slap Bass 2',
    'Synth Bass 1',
    'Synth Bass 2',
    # Strings
    'Violin',
    'Viola',
    'Cello',
    'Contra-Bass',
    'Tremelo-Strings',
    'Pizz.Strings',
    'Orch.Strings',
    'Timpani',
    # Ensemble
    'String-Ens.1',
    'String-Ens.2',
    'Synth.Strings-1',
    'Synth.Strings-2',
    'Choir-Aahs',
    'Voice-Oohs',
    'Synth-Voice',
    'Orchestra-Hit',
    # Brass
    'Trumpet',
    'Trombone',
    'Tuba',
    'Muted-Trumpet',
    'French-Horn',
    'Brass-Section',
    'Synth-Brass-1',
    'Synth-Brass-2',
    # Reed
    'Soprano-Sax',
    'Alto-Sax',
    'Tenor-Sax',
    'Baritone-Sax',
    'Oboe',
    'English-Horn',
    'Bassoon',
    'Clarinet',
    # Pipe
    'Piccolo',
    'Flute',
    'Recorder',
    'Pan-Flute',
    'Blown-Bottle',
    'Shakuhachi',
    'Whistle',
    'Ocarina',
    # Synth Lead
    'Lead1-Square',
    'Lead2-Sawtooth',
    'Lead3-Calliope',
    'Lead4-Chiff',
    'Lead5-Charang',
    'Lead6-Voice',
    'Lead7-Fifths',
    'Lead8-Bass-Ld',
    # Synth Pad
    '9-Pad-1',
    '0-Pad-2',
    '1-Pad-3',
    '2-Pad-4',
    '3-Pad-5',
    '4-Pad-6',
    '5-Pad-7',
    '6-Pad-8',
    # Synth F / X
    'FX1-Rain',
    'FX2-Soundtrack',
    'FX3-Crystal',
    'FX4-Atmosphere',
    'FX5-Brightness',
    'FX6-Goblins',
    'FX7-Echoes',
    'FX8-Sci-Fi',
    # Ethnic
    'Sitar',
    'Banjo',
    'Shamisen',
    'Koto',
    'Kalimba',
    'Bagpipe',
    'Fiddle',
    'Shanai',
    # Percussive
    'TinkerBell',
    'Agogo',
    'SteelDrums',
    'Woodblock',
    'TaikoDrum',
    'Melodic-Tom',
    'SynthDrum',
    'Reverse-Cymbal',
    # Sound F / X
    'Guitar-Fret-Noise',
    'Breath-Noise',
    'Seashore',
    'BirdTweet',
    'Telephone',
    'Helicopter',
    'Applause',
    'Gunshot',
]


class Controller:

    next_midi_output = 0

    def __init__(self, instrument=0):
        self.instrument = self.get_instrument_from_input(instrument)
        self.player = None
        self.get_player()
        self.accepted_keys = ascii_lowercase + "01234567890-=m,./[]'\\"
        self.already_pressed = {k: False for k in self.accepted_keys}

    def get_player(self):
        self.player = pygame.midi.Output(self.next_midi_output)
        self.next_midi_output += 1
        self.player.set_instrument(self.instrument)

    @staticmethod
    def get_instrument_from_input(instrument):
        if isinstance(instrument, int):
            pass
        if isinstance(instrument, str):
            try:
                instrument = int(instrument)
            except ValueError:
                instrument = midi_instruments.index(instrument)
        if not (0 <= instrument <= 127):
            instrument = 0
        return instrument

    def on_press(self, key):
        # print('self', self, 'key', key)
        try:
            # print(f'alphanumeric key {key.char} pressed')
            if not self.already_pressed[key.char]:
                self.player.note_on(key_to_note[key.char], 127)
                self.already_pressed[key.char] = True
        except KeyError:
            pass
        except AttributeError:
            # print(f'special key {key} pressed')
            pass

    def on_release(self, key):
        # print(f'{key} released')
        try:
            self.player.note_off(key_to_note[key.char], 127)
            self.already_pressed[key.char] = False
        except KeyError:
            pass
        except AttributeError:
            pass
        if key == keyboard.Key.esc:
            # Stop listener
            return False


def main(args):
    pygame.midi.init()

    player = pygame.midi.Output(0)
    player.set_instrument(args.inst)
    player.note_on(64, 127)
    time.sleep(1)
    player.note_off(64, 127)
    player.close()

    controller = Controller(instrument=args.inst)

    # Collect events until released
    with keyboard.Listener(
            on_press=controller.on_press,
            on_release=controller.on_release) as listener:
        listener.join()

    '''
    # ...or, in a non-blocking fashion:
    listener = keyboard.Listener(
        on_press=controller.on_press,
        on_release=controller.on_release)
    listener.start()
    '''
    controller.player.close()
    pygame.midi.quit()


def preprocess(args):
    try:
        args.inst = int(args.inst)
    except ValueError:
        args.inst = midi_instruments.index(args.inst)
    return args


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inst', type=str, default='0',
                        help='Number or name of the instrument')
    args = parser.parse_args()
    args =preprocess(args)

    main(args)
