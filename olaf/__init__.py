import pathlib

__version__ = (pathlib.Path(__file__).parents[0] / "VERSION").read_text(
    encoding="utf-8"
)

from .dtypes import *
from ._backends import *

