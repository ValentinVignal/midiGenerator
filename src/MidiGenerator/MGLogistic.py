import os
from termcolor import colored, cprint
from pathlib import Path

from .MGInit import MGInit
from src import Path as mPath


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
        token_path = Path('temp', f'token_{self.full_name}')
        if token_path.exists():
            token_path.unlink()

    def get_new_i(self):
        """

        :return: set up a new unique full name and the corresponding path to save the trained model
        """
        i = 0
        full_name = self.full_name_no_i + '-({0})'
        saved_model_path = Path('saved_models')
        while Path(saved_model_path, full_name.format(i)).exists() or Path('temp',
                                                                           'token_' + full_name.format(i)).exists():
            i += 1
        self.i = i
        print('Got new full_name :', colored(self.full_name, 'blue'))

        self.save_midis_path = None

    def ensure_save_midis_path(self):
        """

        :return:
        """
        if self.save_midis_path is None:
            self.get_new_save_midis_path()

    def get_new_save_midis_path(self, path=None):
        """
        set up a new save Midi path
        :param path:
        :return:
        """
        if path is None:
            path = Path('generated_midis', f'{self.full_name}-generation')
        self.save_midis_path = mPath.new.unique(path, ext='({0})', mandatory_ext=True)
        print('new save path for Midi files :', colored(str(self.save_midis_path), 'cyan'))
