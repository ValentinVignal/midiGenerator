import time
import threading
import numpy as np

from .MidiPlayer import MidiPlayer

from src import GlobalVariables as g


class Metronome(MidiPlayer):
    """
    Class to do metronome while playing. Also trigger the callbacks at the right time
    """
    def __init__(self, tempo, *args,
                 step_length=g.mg.work_on,
                 on_time_step_begin_callbacks=[],
                 on_time_step_end_callbacks=[],
                 on_step_begin_callbacks=[],
                 on_step_end_callbacks=[],
                 on_exact_time_step_begin_callbacks=[],
                 on_exact_time_step_end_callbacks=[],
                 **kwargs):
        super(Metronome, self).__init__(*args, **kwargs)
        self.tempo = tempo
        self.step_length = g.mg.work_on_nb(step_length)  # A number

        self.start_time = None
        self.keep_playing = False
        self.hard_stop = True
        self.thread = None
        self.bip_thread = None

        self.on_time_step_begin_callbacks = on_time_step_begin_callbacks
        self.on_time_step_end_callbacks = on_time_step_end_callbacks
        self.on_step_begin_callbacks = on_step_begin_callbacks
        self.on_step_end_callbacks = on_step_end_callbacks
        self.on_exact_time_step_begin_callbacks = on_exact_time_step_begin_callbacks
        self.on_exact_time_step_end_callbacks = on_exact_time_step_end_callbacks

        self.current_exact_time_step = None     # Exact time step (usually quarter of beat)
        self.current_time_step = None       # Time step rounded to the closest one
        self.current_step = None            # The step (usually measure)
        self.current_beat = None            # The beat (to do tic tac)

    @property
    def current_time_step_mod(self):
        return self.current_time_step % self.step_length

    @property
    def current_beat_mod(self):
        return self.current_beat % 4

    def start(self, start_time=None,
              on_time_step_begin_callbacks=[],
              on_time_step_end_callbacks=[],
              on_step_begin_callbacks=[],
              on_step_end_callbacks=[],
              on_exact_time_step_begin_callbacks=[],
              on_exact_time_step_end_callbacks=[],
              ):
        """

        :param start_time:
        :param on_time_step_begin_callbacks:
        :param on_time_step_end_callbacks:
        :param on_step_begin_callbacks:
        :param on_step_end_callbacks:
        :param on_exact_time_step_begin_callbacks:
        :param on_exact_time_step_end_callbacks:
        :return:
        """
        self.start_time = time.perf_counter() if start_time is None else start_time
        # Callbacks
        self.on_time_step_begin_callbacks.extend(on_time_step_begin_callbacks)
        self.on_time_step_end_callbacks.extend(on_time_step_end_callbacks)
        self.on_step_begin_callbacks.extend(on_step_begin_callbacks)
        self.on_step_end_callbacks.extend(on_step_end_callbacks)
        self.on_exact_time_step_begin_callbacks.extend(on_exact_time_step_begin_callbacks)
        self.on_exact_time_step_end_callbacks.extend(on_exact_time_step_end_callbacks)
        # Start the thread to not block the script
        self.thread = threading.Thread(target=self.play, args=())
        self.thread.start()

    def bip(self, note):
        self.note_on(note)
        time.sleep(self.tempo / (8 * 60))
        self.note_off(note)

    def play(self):
        """

        :return:
        """
        # Reset everything
        self.keep_playing = True        # While true: continue
        self.hard_stop = False          # If we need urgent stop
        current_half_time_step = -1  # Number of the half time step (to be able to round exact time step)
        self.current_time_step = -1  # Number of the time step
        self.current_exact_time_step = -1       # Number of the exact time step
        self.current_beat = -1  # Number of the beat (for bip bip)
        self.current_step = -1  # Number of the step
        while not self.hard_stop:
            t = time.perf_counter()     # Get the time
            half_time_step_number_float = ((t - self.start_time) * self.tempo * g.midi.step_per_beat * 2 / 60)
            half_time_step_number = np.floor(half_time_step_number_float)
            if half_time_step_number > current_half_time_step:      # If this is a new half_time_step
                # Do action for the new time step
                current_half_time_step += 1
                if current_half_time_step % 2 == 0:
                    # Do to the action exactly on time
                    # (Like playing a note)
                    threading.Thread(
                        target=self.call_callbacks,
                        args=(self.on_exact_time_step_end_callbacks, self.current_exact_time_step)
                    ).start()

                    self.current_exact_time_step += 1

                    threading.Thread(
                        target=self.call_callbacks,
                        args=(self.on_exact_time_step_begin_callbacks, self.current_exact_time_step)
                    ).start()

                    # To do Tic tac of metronome
                    beat_number = half_time_step_number // (2 * g.midi.step_per_beat)
                    if beat_number > self.current_beat:
                        self.current_beat += 1
                        note = 72 if self.current_beat % 4 == 0 else 60
                        self.bip_thread = threading.Thread(target=self.bip, args=(note,))
                        self.bip_thread.start()

                else:
                    # Do the action with offset of half a time_step

                    is_new_step = (self.current_time_step // self.step_length) + 1 == (
                                1 + self.current_time_step) // self.step_length

                    # On time step end
                    threading.Thread(
                        target=self.call_callbacks,
                        args=(self.on_time_step_end_callbacks, self.current_time_step)
                    ).start()

                    if is_new_step:
                        # On step end
                        threading.Thread(
                            target=self.call_callbacks,
                            args=(self.on_step_end_callbacks, self.current_step)
                        ).start()

                    self.current_time_step += 1
                    self.current_step = self.current_step + 1 if is_new_step else self.current_step

                    if is_new_step:
                        # On step begin
                        threading.Thread(
                            target=self.call_callbacks,
                            args=(self.on_step_begin_callbacks, self.current_step)
                        ).start()

                    # On time step begin
                    threading.Thread(
                        target=self.call_callbacks,
                        args=(self.on_time_step_begin_callbacks, self.current_time_step)
                    ).start()
                    if not self.keep_playing and is_new_step:
                        break

    @staticmethod
    def call_callbacks(callbacks=[], *args, **kwargs):
        for callback in callbacks:
            callback(*args, **kwargs)

    def stop(self, hard=False):
        """

        :param hard: if False: will finish the last step, if True: won't finish it
        :return:
        """
        self.keep_playing = False
        self.hard_stop = hard
        self.thread.join()
        self.bip_thread.join()
        self.hard_stop = True
