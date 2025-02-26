import pathlib

__version__ = (pathlib.Path(__file__).parents[0] / "VERSION").read_text(
    encoding="utf-8"
)

from .dtypes import *
from ._backends import *
from .autograd import (
    Tensor,
    no_grad,
    tensor,
    arange,
    ones,
    ones_like,
    zeros,
    zeros_like,
    full,
    full_like,
    concat,
    stack,
    where,
)
from .random import *
