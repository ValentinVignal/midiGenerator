"""
To play midi in real time
"""
import pygame.midi

from .. import instruments


class MidiPlayer:
    pygame.midi.init()
    player = pygame.midi.Output(0)

    # Can't have 2 different instruments on the same channel
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
                instrument = instruments.all_midi_instruments.index(instrument)
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
