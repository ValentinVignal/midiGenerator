from pathlib import Path


def unique(path, ext='_({0})', mandatory_ext=False):
    """

    :param mandatory_ext:
    :param path:
    :param ext:
    :return:
    """
    path = Path(path)
    if not path.exists() and not mandatory_ext:
        return path
    else:
        folder_path = path.parent
        suffixes = path.suffixes
        stem = path.stem
        for s in range(len(suffixes) - 1):
            stem = Path(stem).stem
        i = 0
        while (folder_path / ''.join([stem + ext.format(i)] + suffixes)).exists():
            i += 1
        return folder_path / ''.join([stem + ext.format(i)] + suffixes)


def unique_filename(folder_path, name, ext='_({0})', with_extension=True, mandatory_ext=False):
    """

    :param mandatory_ext:
    :param folder_path:
    :param name:
    :param with_extension:
    :return:
    """
    folder_path = Path(folder_path)
    suffixes = Path(name).suffixes
    stem = Path(name).stem
    for s in range(len(suffixes) - 1):
        stem = Path(stem).stem
    if not (folder_path / name).exists() and not mandatory_ext:
        return name if with_extension else stem
    else:
        i = 0
        while (folder_path / ''.join([stem + ext.format(i)] + suffixes)).exists():
            i += 1
        return ''.join([stem + ext.format(i)] + suffixes) if with_extension else stem + ext.format(i)




