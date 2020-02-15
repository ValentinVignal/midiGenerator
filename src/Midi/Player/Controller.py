import numpy as np
import time
from pynput import keyboard

from .MidiPlayer import MidiPlayer
from .Metronome import Metronome
from . import keyboard as kb

from src import GlobalVariables as g


class Controller(MidiPlayer):
    def __init__(self, *args, tempo=120, step_length=g.mg.work_on, **kwargs):
        super(Controller, self).__init__(*args, **kwargs)
        # Raw parameters
        self.tempo = tempo
        self.step_length = g.mg.work_on_nb(step_length)

        # for pygame output
        self.already_pressed = {}

        # metronome
        self.metronome = Metronome(instrument='Woodblock', tempo=self.tempo, step_length=self.step_length)

        # np array
        self.current_array = np.zeros((self.step_length, 128))
        self.arrays = []
        self.current_time_step = 0

    @property
    def current_step(self):
        return self.current_time_step // self.step_length

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
                note = kb.key_to_note[key.char]
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
            self.note_off(kb.key_to_note[key.char])
            self.already_pressed[key.char] = False
        except KeyError:
            pass
        except AttributeError:
            pass
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    def play(self, start_time=None, on_new_step_callbacks=[], on_new_time_step_callback=[]):
        """

        :return:
        """
        start_time = time.perf_counter() if start_time is None else start_time
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.metronome.start(
            start_time=start_time,
            on_new_step_callbacks=[self.on_new_step] + on_new_step_callbacks,
            on_new_time_step_callbacks=[self.on_new_time_step] + on_new_time_step_callback
        )
        # Start of the threads
        listener.start()
        listener.join()
        # Stop of the threads
        self.metronome.stop()

    def on_new_step(self, *args, **kwargs):
        """

        :return:
        """

        self.arrays.append(np.copy(self.current_array))
        self.current_array = np.zeros((self.step_length, 128))

    def on_new_time_step(self, time_step, *args, **kwargs):
        """

        :return:
        """
        self.current_time_step = time_step % self.step_length

    @property
    def array(self):
        return np.concatenate(self.arrays, )
