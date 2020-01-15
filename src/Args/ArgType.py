from enum import Enum


class ArgType(Enum):
    """

    """
    ALL = 0

    Train = 1
    HPSearch = 2
    Generate = 3

    ComputeData = 10
    CheckData = 11
