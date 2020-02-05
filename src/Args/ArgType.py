from enum import Enum


class ArgType(Enum):
    """

    """
    ALL = 0

    Train = 1
    HPSearch = 2
    HPSummary = 3
    Generate = 5

    ComputeData = 10
    CheckData = 11

    Clean = 20
    Zip = 21
