import numpy as np
import time

from .MidiPlayer import MidiPlayer
from .Metronome import Metronome
from . import keyboard

from src import GlobalVariables as g



class Controller(MidiPlayer):
    def __init__(self, *args, tempo=120, work_on=g.mg.work_on, **kwargs):
        super(Controller, self).__init__(*args, **kwargs)
        # Raw parameters
        self.tempo = tempo
        self.work_on = g.mg.work_on_nb(work_on)

        # for pygame output
        self.already_pressed = {}

        # metronome
        self.metronome = None
        self.start_play = None

        # np array
        self.current_array = np.zeros((work_on, 128))
        self.arrays = []
        self.current_time_step = 0

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
                note = keyboard.key_to_note[key.char]
                self.note_on(note)
                self.already_pressed[key.char] = True
                self.current_array[self.current_time_step, note] = 1
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
            self.note_off(keyboard.key_to_note[key.char])
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
        metronome = Metronome(instrument='Woodblock', tempo=self.tempo, work_on=self.work_on)
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        metronome.start(
            start_time=start_time,
            on_new_step_callbacks=[self.on_new_step],
            on_new_time_step_callbacks=[self.on_new_time_step]
        )
        # Start of the threads
        listener.start()
        listener.join()
        # Stop of the threads
        metronome.stop()
        del metronome

    def on_new_step(self, *args, **kwargs):
        """

        :return:
        """

        self.arrays.append(np.copy(self.current_array))
        self.current_array = np.zeros((16, 128))

    def on_new_time_step(self, time_step, *args, **kwargs):
        """

        :return:
        """
        self.current_time_step = time_step % self.work_on

    @property
    def array(self):
        return np.concatenate(self.arrays, )
