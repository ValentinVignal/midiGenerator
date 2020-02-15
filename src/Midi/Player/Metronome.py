import time
import threading
import numpy as np

from .MidiPlayer import MidiPlayer

from src import GlobalVariables as g


class Metronome(MidiPlayer):
    def __init__(self, tempo, *args, step_length=g.mg.work_on, on_new_step_callbacks=[], on_new_time_step_callbacks=[], **kwargs):
        super(Metronome, self).__init__(*args, **kwargs)
        self.tempo = tempo
        self.step_length = g.mg.work_on_nb(step_length)     # A number

        self.start_time = None
        self.keep_playing = False
        self.hard_stop = True
        self.thread = None

        self.on_new_step_callbacks = on_new_step_callbacks
        self.on_new_time_step_callbacks = on_new_time_step_callbacks

        self.current_time_step = None
        self.current_step = None
        self.current_beat = None

    @property
    def current_time_step_mod(self):
        return self.current_time_step % self.step_length

    @property
    def current_beat_mod(self):
        return self.current_beat % 4

    def start(self, start_time=None, on_new_step_callbacks=[], on_new_time_step_callbacks=[]):
        """

        :param start_time:
        :return:
        """
        self.start_time = time.perf_counter() if start_time is None else start_time
        # Callbacks
        self.on_new_step_callbacks.extend(on_new_step_callbacks)
        self.on_new_time_step_callbacks.extend(on_new_time_step_callbacks)
        # Start the thread
        self.thread = threading.Thread(target=self.play, args=())
        self.thread.start()

    def play(self):
        """

        :return:
        """
        half_time_step = -1
        self.keep_playing = True
        self.hard_stop = False
        self.current_time_step = -1
        self.current_beat = -1
        self.current_step = -1
        while not self.hard_stop:
            t = time.perf_counter()
            half_time_step_number_float = ((t - self.start_time) * self.tempo * g.midi.step_per_beat * 2 / 60)
            half_time_step_number = np.floor(half_time_step_number_float)
            if half_time_step_number > half_time_step:
                # Do action for the new time step
                half_time_step += 1
                if half_time_step % 2 == 0:
                    self.current_time_step += 1
                    # Do all the actions exactly on time
                    beat_number = half_time_step_number // (2 * g.midi.step_per_beat)
                    if beat_number > self.current_beat:
                        self.current_beat += 1
                        note = 72 if self.current_beat % 4 == 0 else 60
                        self.note_on(note)
                        time.sleep(self.tempo / (8 * 60))
                        self.note_off(note)
                else:
                    # Do the action with offset of half a time_step
                    for callback in self.on_new_time_step_callbacks:
                        callback((half_time_step // 2) + 1)

                    step_number = half_time_step_number // (2 * self.step_length)
                    if step_number > self.current_step:
                        self.current_step += 1
                        for callback in self.on_new_step_callbacks:
                            callback(self.current_step)
                        if not self.keep_playing:
                            break

    def stop(self, hard=False):
        """

        :param hard: if False: will finish the last step, if True: won't finish it
        :return:
        """
        self.keep_playing = False
        self.hard_stop = hard
        self.thread.join()
        self.hard_stop = True
