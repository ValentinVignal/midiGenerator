from typing import TypeVar, Tuple, Dict, Sequence, Union, Optional, Any, List
import typing
import typeguard

# -------------------- Basics --------------------

NoneType = type(None)
int_ = Optional[int]

# -------------------- Neural Network --------------------

strides = Tuple[int, ...]
strides_ = Optional[strides]

shape = Tuple[int, ...]
shape_ = Optional[shape]

bshape = Tuple[int_, ...]
bshape_ = Optional[bshape]

anyshape = Union[shape, bshape]
anyshape_ = Optional[anyshape]


# ---------- Tensorflow ----------


# ----------------------------------------------------------------------------------------------------
#                                           Functions
# ----------------------------------------------------------------------------------------------------

def check_type(value: Any, expected_type: Any) -> bool:
    try:
        typeguard.check_type('value', value, expected_type)
        return True
    except TypeError:
        return False


# ----------------------------------------------------------------------------------------------------
#                                           Cast
# ----------------------------------------------------------------------------------------------------


class Shape:
    mtype = shape
    mtype_ = shape_

    def __init__(self):
        pass

    @staticmethod
    def cast(value: Any) -> shape:
        if value is None:
            raise TypeError('Can t cast None Value')
        elif check_type(value, shape) or check_type(value, shape_):
            return value
        elif check_type(value, bshape) or check_type(value, bshape_):
            """
            Remove the batch dimension
            """
            return value[1:]

    @staticmethod
    def cast_from(value, type_from):
        if value is None:
            raise TypeError('Can t cast None Value')
        elif type_from is shape or type_from is shape_:
            return value
        elif type_from is bshape or type_from is bshape_:
            """
            Remove the batch dimension
            """
            return value[1:]


class Bshape:
    mtype = bshape
    mtype_ = bshape_

    def __init__(self):
        pass

    @staticmethod
    def cast(value: Any) -> bshape:
        if check_type(value, bshape) or check_type(value, bshape_):
            return value
        elif check_type(value, shape) or check_type(value, shape_):
            """
            Add the batch dimension
            """
            return (None, *value)

    @staticmethod
    def cast_from(value: Any, type_from: Any) -> bshape:
        if type_from is bshape or type_from is bshape_:
            return value
        elif type_from is shape or type_from is shape_:
            """
            Add the batch dimension
            """
            return (None, *value)


class MyTypes:
    def __init__(self):
        self.shape = Shape()
        self.bshape = Bshape()


m_types = MyTypes()
