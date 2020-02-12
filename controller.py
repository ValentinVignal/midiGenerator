from pynput import keyboard
import pygame.midi
from argparse import ArgumentParser
import time
import _thread
import threading

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


class MidiPlayer:
    pygame.midi.init()
    player = pygame.midi.Output(0)

    taken_channels = [False for _ in range(16)]

    def __init__(self, instrument=0):
        self.instrument = self.get_instrument_from_input(instrument)

        self.channel = None
        self.get_channel()

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

    def get_channel(self):
        for i in range(16):
            if not MidiPlayer.taken_channels[i]:
                self.channel = i
                MidiPlayer.taken_channels[i] = True
                MidiPlayer.player.set_instrument(self.instrument, self.channel)
                break

    def __del__(self):
        MidiPlayer.taken_channels[self.channel] = False

    def note_on(self, note, velocity=127):
        MidiPlayer.player.note_on(note, velocity, self.channel)

    def note_off(self, note, velocity=127):
        MidiPlayer.player.note_off(note, velocity, self.channel)


class Metronome(MidiPlayer):
    def __init__(self, *args, tempo, **kwargs):
        super(Metronome, self).__init__(*args, **kwargs)
        self.tempo = tempo

        self.start_time = None
        self.keep_playing = False
        self.thread = None

    def start(self, start_time=None):
        """

        :param start_time:
        :return:
        """
        self.start_time = time.perf_counter() if start_time is None else start_time
        """
        try:
            _thread.start_new_thread(self.play)
        except:
            print('Cannot start metronome')
        """

        self.thread = threading.Thread(target=self.play, args=())
        self.thread.start()

    def play(self):
        """

        :return:
        """
        beat = -1
        self.keep_playing = True
        while self.keep_playing:
            if ((time.perf_counter() - self.start_time) * self.tempo) // 60  > beat:
                beat += 1
                note = 72 if beat % 4 == 0 else 60
                self.note_on(note)
                time.sleep(self.tempo / (8 * 60))
                self.note_off(note)

    def stop(self):
        self.keep_playing = False
        self.thread.join()


class Controller(MidiPlayer):
    def __init__(self, *args, tempo=120, **kwargs):
        super(Controller, self).__init__(*args, **kwargs)
        # Raw parameters
        self.tempo = tempo

        # for pygame output
        self.already_pressed = {}

        # metronome
        self.metronome = None
        self.start_play = None

    def on_press(self, key):
        """

        :param key:
        :return:
        """
        # print('self', self, 'key', key)
        try:
            # print(f'alphanumeric key {key.char} pressed')
            if not key.char in self.already_pressed:
                self.already_pressed[key.char] = False
            if not self.already_pressed[key.char]:
                self.note_on(key_to_note[key.char])
                self.already_pressed[key.char] = True
        except KeyError:
            pass
        except AttributeError:
            # print(f'special key {key} pressed')
            pass

    def on_release(self, key):
        """

        :param key:
        :return:
        """
        # print(f'{key} released')
        try:
            self.note_off(key_to_note[key.char])
            self.already_pressed[key.char] = False
        except KeyError:
            pass
        except AttributeError:
            pass
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    def play(self):
        """

        :return:
        """
        start_time = time.perf_counter()
        metronome = Metronome(instrument='Woodblock', tempo=self.tempo)
        metronome.start(start_time=start_time)
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        ) as listener:
            listener.join()
        metronome.stop()
        del metronome


def main(args):
    """
    pygame.midi.init()

    player = pygame.midi.Output(0)
    player.set_instrument(args.inst)
    player.note_on(64, 127)
    time.sleep(1)
    player.note_off(64, 127)
    player.close()
    """

    controller = Controller(instrument=args.inst, tempo=args.tempo)
    controller.play()

    '''
    # Collect events until released
    with keyboard.Listener(
            on_press=controller.on_press,
            on_release=controller.on_release) as listener:
        listener.join()

    # ...or, in a non-blocking fashion:
    listener = keyboard.Listener(
        on_press=controller.on_press,
        on_release=controller.on_release)
    listener.start()
    '''
    MidiPlayer.player.close()
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
    parser.add_argument('--tempo', type=int, default=120,
                        help='Tempo')
    args = parser.parse_args()
    args =preprocess(args)

    main(args)
