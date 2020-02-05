import skopt
from .Dimensions import Dimensions


def from_checkpoint(path):
    """

    :param path:
    :return:
    """
    search_result = skopt.load(path / 'search_result.pkl')
    dimensions = Dimensions()
    dimensions.load(path / 'dimensions.p')
    return search_result, dimensions



