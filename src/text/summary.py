from pathlib import Path
from termcolor import cprint
import matplotlib.pyplot as plt


def summarize(path, title=None, file_name='Summary', **d):
    """

    :param path: The folder where to save
    :param title: Title of the summary
    :param file_name: Name of the file (without .txt)
    :param d: All the parameters to put in the summary
    :return:
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    text = ''
    if title is not None:
        text += f'\t\t{title}\n\n'
    for key in d:
        text += f'{key} : {d[key]}\n'
    file_path = (path / file_name).with_suffix('.txt')
    if file_path.exists():
        file_path.unlink()
    with open(str(file_path), 'a') as f:
        f.write(text)


