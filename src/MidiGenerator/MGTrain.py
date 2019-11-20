from termcolor import colored, cprint

from src.NN.sequences.MissingInstSequence import MissingInstSequence
from src.NN.sequences.AllInstSequence import AllInstSequence
import src.global_variables as g
import src.image.pianoroll as pianoroll


class MGTrain:

    def train(self, epochs=None, batch=None, callbacks=[], verbose=1, noise=g.noise, validation=0.0):
        """

        :param epochs:
        :param batch:
        :param callbacks:
        :param verbose:
        :param noise:
        :param validation:
        :return: train the model
        """

        # Do we have to create a new MySequence Object ?
        flag_new_sequence = False
        epochs = 50 if epochs is None else epochs
        if batch is None and self.batch is None:
            self.batch = 1,
            flag_new_sequence = True
        if batch is not None and batch != self.batch:
            self.batch = batch
            flag_new_sequence = True
        if self.my_sequence is None:
            flag_new_sequence = True

        if flag_new_sequence:
            self.my_sequence = MissingInstSequence(
                path=str(self.data_transformed_pathlib),
                nb_steps=int(self.model_id.split(',')[2]),
                batch_size=self.batch,
                work_on=self.work_on
            )
        if noise is not None:
            self.my_sequence.set_noise(noise)

        # Actual train
        print(colored('Training...', 'blue'))
        self.train_history = self.keras_nn.train_seq(epochs=epochs, generator=self.my_sequence, callbacks=callbacks,
                                                     verbose=verbose, validation=validation)

        # Update parameters
        self.total_epochs += epochs
        self.get_new_full_name()
        print(colored('Training done', 'green'))

    # --------------------------------------------------
    #                Test the model
    # --------------------------------------------------

    def evaluate(self, batch=None):
        if batch is not None:
            self.batch = batch
        if self.batch is None:
            self.batch = 4
        cprint('Evaluation', 'blue')
        if self.my_sequence is None:
            self.my_sequence = AllInstSequence(
                path=str(self.data_transformed_pathlib),
                nb_steps=int(self.model_id.split(',')[2]),
                batch_size=self.batch,
                work_on=self.work_on
            )
        evaluation = self.keras_nn.evaluate(generator=self.my_sequence)

        metrics_names = self.keras_nn.model.metrics_names
        text = ''
        for i in range(len(metrics_names)):
            text += metrics_names[i] + ' ' + colored(evaluation[i], 'magenta') + ' -- '
        print(text)

    def test_on_batch(self, i=0, batch_size=4):
        self.my_sequence.change_batch_size(batch_size=batch_size)
        x, y = self.my_sequence[i]
        evaluation = self.keras_nn.model.test_on_batch(x=x, y=y, sample_weight=None)

        metrics_names = self.keras_nn.model.metrics_names
        text = ''
        for i in range(len(metrics_names)):
            text += metrics_names[i] + ' ' + colored(evaluation[i], 'magenta') + ' -- '
        print(text)

    def predict_on_batch(self, i, batch_size=4):
        self.my_sequence.change_batch_size(batch_size=batch_size)
        x, y = self.my_sequence[i]
        evaluation = self.keras_nn.model.predict_on_batch(x=x)

        return evaluation

    def compare_test_predict_on_batch(self, i, batch_size=4):
        print('compare test predict on batch')
        self.test_on_batch(i, batch_size=batch_size)
        x, yt = self.my_sequence[i]
        yp = self.predict_on_batch(i, batch_size=batch_size)
        pianoroll.see_compare_on_batch(x, yt, yp)

