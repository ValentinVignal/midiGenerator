
from termcolor import colored, cprint

from src import GlobalVariables as g
import src.image.pianoroll as pianoroll
from .MGInit import MGInit


class MGTrain(MGInit):

    def train(self, epochs=None, batch=None, callbacks=[], verbose=1, noise=g.train.noise, validation=0.0,
              sequence_to_numpy=False, fast_sequence=False, memory_sequence=False):
        """

        :param memory_sequence:
        :param fast_sequence:
        :param sequence_to_numpy:
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
        if self.sequence is None:
            flag_new_sequence = True

        if flag_new_sequence:
            self.get_sequence(
                path=str(self.data_transformed_path),
                nb_steps=int(self.model_id.split(',')[2]),
                batch_size=self.batch,
                work_on=self.work_on,
                predict_offset=self.predict_offset,
            )
        if noise is not None:
            self.sequence.set_noise(noise)

        # Actual train
        print(colored('Training...', 'blue'))
        self.train_history = self.keras_nn.train_seq(epochs=epochs, generator=self.sequence, callbacks=callbacks,
                                                     verbose=verbose, validation=validation,
                                                     sequence_to_numpy=sequence_to_numpy, fast_seq=fast_sequence,
                                                     memory_seq=memory_sequence)

        # Update parameters
        self.total_epochs += epochs
        self.get_new_full_name()
        print(colored('Training done', 'green'))
        return self.train_history

    # --------------------------------------------------
    #                Test the model
    # --------------------------------------------------

    def evaluate(self, batch=None):
        if batch is not None:
            self.batch = batch
        if self.batch is None:
            self.batch = 4
        cprint('Evaluation', 'blue')
        if self.sequence is None:
            self.get_sequence(
                path=str(self.data_transformed_path),
                nb_steps=int(self.model_id.split(',')[2]),
                batch_size=self.batch,
                work_on=self.work_on
            )
        evaluation = self.keras_nn.evaluate(generator=self.sequence)

        metrics_names = self.keras_nn.model.metrics_names
        text = ''
        for i in range(len(metrics_names)):
            text += metrics_names[i] + ' ' + colored(evaluation[i], 'magenta') + ' -- '
        print(text)

    def test_on_batch(self, i=0, batch_size=4):
        self.sequence.change_batch_size(batch_size=batch_size)
        x, y = self.sequence[i]
        evaluation = self.keras_nn.model.test_on_batch(x=x, y=y, sample_weight=None)

        metrics_names = self.keras_nn.model.metrics_names
        text = ''
        for i in range(len(metrics_names)):
            text += metrics_names[i] + ' ' + colored(evaluation[i], 'magenta') + ' -- '
        print(text)

    def predict_on_batch(self, i, batch_size=4):
        self.sequence.change_batch_size(batch_size=batch_size)
        x, y = self.sequence[i]
        evaluation = self.keras_nn.model.predict_on_batch(x=x)

        return evaluation

    def compare_test_predict_on_batch(self, i, batch_size=4):
        print('compare test predict on batch')
        self.test_on_batch(i, batch_size=batch_size)
        x, yt = self.sequence[i]
        yp = self.predict_on_batch(i, batch_size=batch_size)
        pianoroll.see_compare_on_batch(x, yt, yp)
