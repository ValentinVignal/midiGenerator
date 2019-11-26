import os
from termcolor import colored, cprint
from pathlib import Path

from .MGInit import MGInit


class MGLogistic(MGInit):

    # --------------------------------------------------
    #                   Names function
    # --------------------------------------------------

    def get_full_name(self, i):
        """

        :param i: index
        :return: set up the full name and the path to save the trained model
        """
        full_name = f'{self.name}-m({self.model_id})-wo({self.work_on})-e({self.total_epochs})-({i})'
        saved_model_path = Path('saved_models') / full_name
        self.full_name = full_name
        self.saved_model_path = saved_model_path
        print('Get full_name :', colored(self.full_name, 'blue'))

    def get_new_full_name(self):
        """

        :return: set up a new unique full name and the corresponding path to save the trained model
        """
        i = 0
        full_name = f'{self.name}-m({self.model_id})-wo({self.work_on})-e({self.total_epochs})-({i})'
        saved_model_path = Path(os.path.join('saved_models', full_name))
        while saved_model_path.exists():
            i += 1
            full_name = f'{self.name}-m({self.model_id})-wo({self.work_on})-e({self.total_epochs})-({i})'
            saved_model_path = Path('saved_models', full_name)
        self.saved_model_path = saved_model_path
        self.full_name = full_name
        print('Got new full_name :', colored(self.full_name, 'blue'))

        self.save_midis_path = None

    def set_name(self, name=None):
        """

        :param name:
        :return:
        """
        self.name = self.name if name is None else name
        self.get_new_full_name()

    def get_new_save_midis_path(self, path=None):
        """
        set up a new save Midi path
        :param path:
        :return:
        """
        if path is None:
            i = 0
            m_str = f'{self.full_name}-generation({i})'
            while Path('generated_midis', m_str).exists():
                i += 1
                m_str = f'{self.full_name}-generation({i})'
            self.save_midis_path = Path('generated_midis', m_str)
        else:
            self.save_midis_path = Path(path)
        print('new save path for Midi files :', colored(str(self.save_midis_path), 'cyan'))

