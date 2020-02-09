from termcolor import colored, cprint
from pathlib import Path

from .MGInit import MGInit


class MGLogistic(MGInit):

    # --------------------------------------------------
    #                   Names function
    # --------------------------------------------------

    def create_token(self):
        """
        create a token in the temp file for the save folder
        :return:
        """
        Path('temp').mkdir(exist_ok=True, parents=True)
        with open(Path('temp', 'token_' + self.full_name), 'w') as f:
            f.write(f'token for the model {self.full_name}')

    def delete_token(self):
        """
        Delete the token associated to the model
        :return:
        """
        if self.full_name_i is not None:
            token_path = Path('temp', f'token_{self.full_name}')
            if token_path.exists():
                token_path.unlink()

    def get_new_i(self):
        """

        :return: set up a new unique full name and the corresponding path to save the trained model
        """
        self.delete_tokens()
        i = 0
        full_name = self.full_name_no_i + '-({0})'
        saved_model_path = Path('saved_models')
        while Path(saved_model_path, full_name.format(i)).exists() or Path('temp',
                                                                           'token_' + full_name.format(i)).exists():
            i += 1
        self.full_name_i = i
        print('Got new full_name :', colored(self.full_name, 'blue'))
        self.get_new_save_midis_path_i()

    def get_new_save_midis_path_i(self):
        """
        set up a new save Midi path
        :return:
        """
        self.delete_token_midis_path()
        i = 0
        name = self.full_name + '-generation({0})'
        while Path('generated_midis', name.format(i)).exists() or Path('temp',
                                                                       ('token_' + name + '.txt').format(i)).exists():
            i += 1
        self.save_midis_path_i = i
        print('new save path for Midi files :', colored(str(self.save_midis_path), 'cyan'))

    def create_token_midis_path(self):
        """

        :return:
        """
        Path('temp').mkdir(exist_ok=True, parents=True)
        with open(Path('temp', 'token_' + self.full_name + f'-generation({self.save_midis_path_i}).txt'), 'w') as f:
            f.write(f'token for the generation folder at {self.save_midis_path}')

    def delete_token_midis_path(self):
        """

        :return:
        """
        if self._save_midis_path_i is not None:
            path = Path('temp', f'token_{self.full_name}-generation({self._save_midis_path_i}).txt')
            if path.exists():
                path.unlink()

    def delete_tokens(self):
        """

        :return:
        """
        self.delete_token_midis_path()
        self.delete_token()

    def create_tokens(self):
        """

        :return:
        """
        self.create_token()
        self.create_token_midis_path()



