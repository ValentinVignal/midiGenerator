from .KerasLayer import KerasLayer
from src.NN import Loss

from src import GlobalVariables as g


class Harmony(KerasLayer):
    def __init__(self, *args, l_semitone=g.loss.l_semitone, l_tone=g.loss.l_tone, l_tritone=g.loss.l_tritone,
                 mono=False, **kwargs):
        """

        :param args:
        :param l_semitone:
        :param l_tone:
        :param l_tritone:
        :param mono:
        :param kwargs:
        """
        super(Harmony, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.l_semitone = l_semitone,
        self.l_tone = l_tone,
        self.l_tritone = l_tritone
        self.mono = mono

        self.f = Loss.cost.harmony(l_semitone=self.l_semitone, l_tone=self.l_tone, l_tritone=self.l_tritone)

    def get_config(self):
        config = super(Harmony, self).get_config()
        config.update(
            l_semitone=self.l_semitone,
            l_tone=self.l_tone,
            l_tritone=self.l_tritone,
            mono=self.mono
        )
        return config

    def build(self, inputs_shape):
        super(Harmony, self).build(inputs_shape)

    def call(self, inputs):
        """

        :param inputs: (alloutput): (batch, nb_instruments, nb_steps, step_size, input_size, channels)
        :return:
        """
        inputs_a = Loss.utils.get_activation(inputs)
        if self.mono:
            inputs_a = inputs_a[:, :, :, :, :-1]
        return self.f(inputs_a)

    def compute_output_shape(self, input_shape):
        return 0,


