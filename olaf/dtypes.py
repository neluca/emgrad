"""Array data types."""

from typing import Optional, TypeAlias

import numpy as np

__all__ = [
    "Scalar",
    "ArrayLike",
    "Dim",
    "ShapeLike",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool_",
]

Scalar: TypeAlias = int | float
ArrayLike: TypeAlias = np.ndarray
Dim = int | tuple[int, ...]

DType: TypeAlias = type


class Shape(tuple):
    """Array shape as a tuple of integers."""

    def __repr__(self) -> str:
        s = super().__repr__().replace("(", "").replace(")", "")
        return f"shape({s})"


ShapeLike: TypeAlias = Shape | tuple[int, ...]

# Float types
float16 = np.float16
float32 = np.float32
float64 = np.float64

# Integer types
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

# Unsigned integer types
uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

# Boolean type
bool_ = np.bool_


def select_dtype(dtype: Optional[DType]) -> DType:
    """Selects a dtype, defaulting to `float32` if none is provided.

    Args:
        dtype (DType | None): The data type to select.

    Returns:
        DType: The selected data type, defaulting to `float32`.
    """
    return dtype or float32


def is_float(dtype: DType) -> bool:
    """Checks if a given data type is a floating-point type.

    Args:
        dtype (DType): The data type to check.

    Returns:
        bool: True if the dtype is `float16`, `float32`, or `float64` otherwise, `False`.
    """
    return any(dtype == d for d in [float16, float32, float64])


def is_int(dtype: DType) -> bool:
    """Checks if a given data type is a integer type.

    Args:
        dtype (DType): The data type to check.

    Returns:
        bool: True if the dtype is `int16`, `int32`, or `int64` otherwise, `False`.
    """
    return any(dtype == d for d in [int16, int32, int64])
