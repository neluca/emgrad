from typing import Optional, TypeAlias
import numpy as np

__all__ = [
    "Scalar",
    "ArrayLike",
    "Dim",
    "DType",
    "Shape",
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
    "select_dtype",
    "is_float",
    "is_integer"
]

Scalar: TypeAlias = int | float
ArrayLike: TypeAlias = np.ndarray
Dim = int | tuple[int, ...]

DType: TypeAlias = type


class Shape(tuple):
    def __repr__(self) -> str:
        return f"shape{super().__repr__()}"


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
    return dtype or float32


def is_float(dtype: DType) -> bool:
    return any(dtype == d for d in [float16, float32, float64])


def is_integer(dtype: DType) -> bool:
    return any(dtype == d for d in [int16, int32, int64])
