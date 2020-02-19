import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp


class RealTimePianoroll:
    """
    This class is used to plot some pianoroll to the user as fast as possible

    """
    def __init__(self, notes_range=None):
        """

        :param notes_range: note range 0 <= x <= 88
                0 : A
                87 : C
        """
        self.piano = None if notes_range is None else self.get_piano(notes_range)
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter()
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    @staticmethod
    def get_piano(notes_range):
        piano = 255 * np.ones((notes_range[1] - notes_range[0], 4, 3), dtype=np.int)       # (input_size, l, 3)
        for note in range(*notes_range):
            if note % 12 in [1, 4, 6, 9, 11]:   # (A#, C#, D#, F#, G#)
                piano[note - notes_range[0]] = 0
        return np.flip(piano, axis=0)

    def show_pianoroll(self, pianoroll):
        if self.piano is not None:
            pianoroll = np.concatenate([self.piano, pianoroll], axis=1)
        self.plot_pipe.send(pianoroll)

    def __del__(self):
        self.plot_pipe.send(None)


class ProcessPlotter:
    def __init__(self):
        self.array = np.zeros((10, 10, 3)).astype(np.int)

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.array = command
                self.ax.imshow(self.array)
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()

        plt.show()
