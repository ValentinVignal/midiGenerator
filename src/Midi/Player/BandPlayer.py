import numpy as np

from .Controller import Controller


class BandPlayer(Controller):
    """

    """

    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.mask = self.model.get_mask()

        super(BandPlayer, self).__init__(*args, step_length=self.model.work_on, **kwargs)

        self.arrays.extend([np.zeros((self.step_length, 128)) for _ in range(self.model.nb_steps)])

        self.model_outputs = [
            np.zeros((self.model.nb_instruments, 1, 1, self.step_length, self.model.input_size, 1)) for _ in
            range(self.model.nb_steps)
        ]  # Actually, self.model.nb_steps - 1 would be enough
        # (nb_instrument, batch=1, nb_step=1, step_length, input_size, channels=1)
        # What is currently played by the model
        self.current_model_part = np.zeros((self.model.nb_instruments, 1, 1, self.step_length, self.model.input_size, 1))
        # What will be played by the model on the next step
        self.next_model_part = np.zeros((self.model.nb_instruments, 1, 1, self.step_length, self.model.input_size, 1))

    def play(self):
        """

        :return:
        """
        super(BandPlayer, self).play(
            on_new_step_callbacks=[self.feed_model],
            on_new_time_step_callback=[]
        )

    def feed_model(self, *args, **kwargs):
        """
        to be run on step_end

        :return:
        """
        self.model_outputs.append(self.current_model_part)
        self.current_model_part = self.next_model_part
        inputs_models = np.concatenate(self.model_outputs[-self.model.nb_steps:], axis=2)
        # input_models: (nb_instruments, batch=1, nb_steps, step_length, input_size, channels)
        played_inst_input = np.concatenate(
            self.arrays[-self.model.nb_steps:], axis=0
        )[:, self.model.notes_range[0]: self.model.notes_range[1]]
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
        inputs_models[0] = played_inst_input
        outputs_model = self.model.keras_nn.generate(input=list(inputs_models) + self.mask)
        # output_model: (nb_instruments, batch=1, np_steps=1, step_length, input_size, channels)
        self.next_model_part = np.asarray(outputs_model)
