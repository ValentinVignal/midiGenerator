import numpy as np

from .Controller import Controller
from .MidiPlayer import MidiPlayer
from ..create import normalize_activation
from ..open import note_to_midinote
from ..instruments import music21_instruments_dict
from src import Images


class BandPlayer(Controller):
    """

    """

    def __init__(self, model,
                 instrument=None, played_voice=0, include_output=True, instrument_mask=None, max_plotted=None,
                 *args,
                 **kwargs):
        """

        :param model: The model to generate the notes
        :param instrument: The instrument the player wants to play, if None, it will be the one of the voice played of
                            the model
        :param played_voice: The voice the player wants to played < model.nb_instruments
        :param include_output: Whether the model should use its previous outputs as a input
        :param instrument_mask: Whether we want to mask some instruments, If not provided, nothing if masked
        :param args:
        :param kwargs:
        """
        # ---------- Raw parameters ----------
        self.model = model
        self.played_voice = played_voice
        self.include_output = include_output
        self.nn_mask = self.get_nn_mask(instrument_mask=instrument_mask)  # (batch=1, nb_instruments, nb_steps)
        self.instrument_mask = [1 for _ in
                                range(self.model.nb_instruments)] if instrument_mask is None else instrument_mask
        self.max_plotted = model.nb_steps if max_plotted is None else max_plotted

        self.band_players = []
        self.set_band_players(instrument=instrument, played_voice=played_voice)
        instrument = self.band_players[played_voice].instrument if instrument is None else instrument

        # ---------- Do super call ----------
        super(BandPlayer, self).__init__(*args, instrument=instrument, step_length=self.model.work_on, **kwargs)
        # Overwrite the instrument of MidiPlayer
        self.arrays.extend([np.zeros((self.step_length, 128)) for _ in range(self.model.nb_steps)])

        # ---------- Create the objects to save what it producing the model ----------
        self.model_outputs = [
            np.zeros((self.model.nb_instruments, 1, 1, self.step_length, self.model.input_size, 1)) for _ in
            range(self.model.nb_steps)
        ]  # Actually, self.model.nb_steps - 1 would be enough
        # (nb_instrument, batch=1, nb_step=1, step_length, input_size, channels=1)
        # What is currently played by the model
        self.current_model_part = np.zeros(
            (self.model.nb_instruments, 1, 1, self.step_length, self.model.input_size, 1))
        # What will be played by the model on the next step
        self.next_model_part = np.zeros((self.model.nb_instruments, 1, 1, self.step_length, self.model.input_size, 1))

        # To know what note we have to stop when a new not is played
        self.previous_notes = [None for _ in range(self.model.nb_instruments)]
        # to be able to switch between 128 midi note to the ones used by the model
        self.midi_notes_range = (note_to_midinote(self.model.notes_range[0]),
                                 note_to_midinote(self.model.notes_range[1]))

        # ---------- To plot the played notes ----------
        self.real_time_pianoroll = Images.RealTimePianoroll(self.model.notes_range)

    def set_band_players(self, instrument=None, played_voice=0):
        """

        :param instrument:
        :param played_voice:
        :return:
        """
        self.band_players = []
        for i_inst in range(self.model.nb_instruments):
            instrument = music21_instruments_dict[self.model.instruments[i_inst]]().midiProgram
            self.band_players.append(MidiPlayer(instrument=instrument))

    def get_nn_mask(self, instrument_mask=None):
        nn_mask = self.model.get_mask()  # (batch=1, nb_instruments, nb_steps)
        if instrument_mask is not None:
            for i in range(self.model.nb_instruments):
                nn_mask[0][:, i] = instrument_mask[i]
        return nn_mask

    def play(self, on_step_end_callbacks=[], on_exact_time_step_begin_callbacks=[], on_time_step_end_callbacks=[],
             **kwargs):
        """

        :return:
        """
        super(BandPlayer, self).play(
            on_time_step_end_callbacks=on_time_step_end_callbacks,
            on_step_end_callbacks=[self.feed_model, self.show_pianoroll_beat] + on_step_end_callbacks,
            on_exact_time_step_begin_callbacks=[self.play_current_model_part] + on_exact_time_step_begin_callbacks,
            **kwargs
        )

    def show_pianoroll_beat(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        played_model = self.get_played_model()  # (nb_instruments, input_size, length, 1)
        arr_pianoroll = Images.pianoroll.array_to_pianoroll(
            array=played_model,
            seed_length=self.step_length,
            mono=False,
            replicate=True,
            colors=None
        )
        self.real_time_pianoroll.show_pianoroll(arr_pianoroll.astype(np.int))

    def get_played_model(self):
        """

        :return: nb_instruments, input_size, length, 1)
        """
        # self.model_outputs[i]: (nb_instruments, batch=1, nb_steps=1, step_length, input_size, channels)
        played_model = np.concatenate(self.model_outputs[-self.max_plotted:], axis=3)
        # played_model: (nb_instruments, batch=1, nb_steps=1, length, input_size, channels)
        played_model = played_model[:, 0, 0, :, :, 0:1]  # (nb_instruments, length, input_size, 1)
        if self.model.mono:
            played_model = played_model[:, :, :-1, :]
        played_inputs = np.concatenate(self.arrays[-self.max_plotted:], axis=0)[:, :, np.newaxis]
        # played_inputs: (length, 128, 1)
        played_model[self.played_voice] = played_inputs[:, self.midi_notes_range[0]: self.midi_notes_range[1], :]
        # played_model: (nb_instruments, length, input_size, 1)
        played_model = np.transpose(played_model, axes=[0, 2, 1, 3])
        # played_model: (nb_instruments, input_size, length, 1)
        played_model = self.apply_mask_on_array(played_model)
        return played_model

    def apply_mask_on_array(self, array):
        """

        :param array:
        :return:
        """
        for inst, m in enumerate(self.instrument_mask):
            if m == 0:
                array[inst] = 0
        return array

    def show_pianoroll_time_step(self, time_step, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        time_step = time_step % self.step_length
        played_model = self.get_played_model()  # (nb_instruments, input_size, length, 1)
        current_played_model = self.current_model_part[:, 0, 0, :time_step + 1, :, 0:1]
        # current_played_model: (nb_instruments, time_step, input_size, 1)
        current_played_model = self.apply_mask_on_array(current_played_model)
        if self.model.mono:
            current_played_model = current_played_model[:, :, :-1, :]
        current_input_played = self.current_array[
                               :time_step + 1, self.midi_notes_range[0]: self.midi_notes_range[1], None
                               ]  # (length, input_size, 1)
        current_played_model[self.played_voice] = current_input_played
        current_played_model = np.transpose(current_played_model, [0, 2, 1, 3])
        # current_played_model: (nb_instrument, input_size, time_step, 1)
        played_model = np.concatenate([played_model, current_played_model], axis=2)
        arr_pianoroll = Images.pianoroll.array_to_pianoroll(
            array=played_model,
            seed_length=self.step_length,
            mono=False,
            replicate=True,
            colors=None
        )[:, -self.max_plotted * self.step_length:, :]  # (input_size, length, 3)
        self.real_time_pianoroll.show_pianoroll(arr_pianoroll.astype(np.int))

    def feed_model(self, *args, **kwargs):
        """
        to be run on step_end

        :return:
        """
        self.model_outputs.append(self.current_model_part)
        self.current_model_part = self.next_model_part
        inputs_models = np.concatenate(self.model_outputs[-self.model.nb_steps:], axis=2)
        if not self.include_output:
            inputs_models = np.zeros_like(inputs_models)
        # input_models: (nb_instruments, batch=1, nb_steps, step_length, input_size, channels)
        played_inst_input = np.concatenate(
            self.arrays[-self.model.nb_steps:], axis=0
        )[:, self.midi_notes_range[0]: self.midi_notes_range[1]]
        # played_inst_input: (nb_steps * step_length, input_size)
        if self.model.mono:
            # We have to had the note "no note"
            no_note_raw = 1 * (np.sum(played_inst_input, axis=1, keepdims=True) == 0)
            # no_note_raw: Should be an array of shape (nb_steps * step_length, 1) with 1 when no note as been played at
            # this time step, and 0 if a note has already been played
            played_inst_input = np.concatenate([played_inst_input, no_note_raw], axis=1)
        played_inst_input = np.reshape(played_inst_input,
                                       newshape=(1, self.model.nb_steps, self.step_length, self.model.input_size, 1))
        # played_inst_input: (batch=1, nb_steps, step_length, input_size, 1
        inputs_models[self.played_voice] = played_inst_input
        outputs_model = self.model.keras_nn.generate(input=list(inputs_models) + self.nn_mask)
        outputs_model = np.asarray(outputs_model).astype('float64')

        # output_model: (nb_instruments, batch=1, np_steps=1, step_length, input_size, channels)
        self.next_model_part = normalize_activation(outputs_model, mono=self.model.mono)

    def play_current_model_part(self, exact_time_step):
        """
        Plays the models notes

        :param time_step:
        :return:
        """
        exact_time_step = exact_time_step % self.step_length
        for instrument in range(0, self.model.nb_instruments):
            if self.instrument_mask[instrument] == 1 and instrument != self.played_voice:
                # current_model_part: (nb_instruments, batch=1, nb_steps=1, step_length, input_size, channels)
                current_instrument_step = self.current_model_part[instrument, 0, 0, exact_time_step, :, 0]
                # current_instrument_step: (input_size)
                current_instrument_step = current_instrument_step[:-1] if self.model.mono else current_instrument_step
                if np.any(current_instrument_step):
                    note = np.where(current_instrument_step == 1)[0][0]
                    note = note_to_midinote(note, notes_range=self.model.notes_range)
                    if self.previous_notes[instrument] is not None:
                        self.band_players[instrument].note_off(self.previous_notes[instrument])
                    self.band_players[instrument].note_on(note)
                    self.previous_notes[instrument] = note
