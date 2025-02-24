"""Computation backends."""

import re
from typing import Any, Optional, TypeAlias
from .dtypes import ArrayLike
import numpy

__all__ = [
    "Device",
    "DeviceLike",
    "get_available_devices",
    "get_array_device",
    "set_random_seed",
    "parse_device",
    "move_to_device"
]

_MAX_LINE_WIDTH = 200
_PRECISION = 4
_FLOATMODE = "maxprec_equal"

_CPU_BACKEND = numpy
_CPU_BACKEND.set_printoptions(precision=_PRECISION, linewidth=_MAX_LINE_WIDTH, floatmode=_FLOATMODE)

try:
    import cupy

    _GPU_BACKEND = cupy
    _GPU_BACKEND.set_printoptions(precision=_PRECISION, linewidth=_MAX_LINE_WIDTH, floatmode=_FLOATMODE)
except ImportError:
    _GPU_BACKEND = None


def gpu_available():
    """Checks if a GPU is available.

    Returns:
        bool: `True` if a CUDA-compatible GPU is available, otherwise `False`.
    """
    return _GPU_BACKEND is not None and _GPU_BACKEND.cuda.is_available()


def set_random_seed(seed: int):
    """Sets the random seed for reproducibility on all devices.

    Args:
        seed (int): The seed value to set.
    """
    _CPU_BACKEND.random.seed(seed)
    if gpu_available():
        _GPU_BACKEND.random.seed(seed)


def array_to_string(data: ArrayLike, prefix: str) -> str:
    """Converts an array to a formatted string.

    Args:
        data (ArrayLike): The array to convert.
        prefix (str): A prefix for formatting the output.

    Returns:
        str: A string representation of the array.
    """
    device = get_array_device(data)
    return device.xp.array2string(
        data,
        max_line_width=_MAX_LINE_WIDTH,
        precision=_PRECISION,
        separator=", ",
        prefix=prefix,
        floatmode=_FLOATMODE,
    )


def _get_type_and_id(device_type: str) -> tuple[str, Optional[int]]:
    match = re.match(r"(?P<type>cpu|cuda)(?::(?P<id>\d+))?", device_type)
    if match:
        device_type = match.group("type")
        if device_type == "cuda":
            assert gpu_available(), "GPUs are not available."
        device_id = match.group("id")
        return device_type, None if device_id is None else int(device_id)
    raise ValueError(f"Unknown device: {device_type}")


class Device:
    """Represents a computing device.

    Args:
        dev_type (str): The type and optionally the id of device (e.g. "cpu" or "cuda:0").
    """

    def __init__(self, dev_type: str):
        dev_type, dev_id = _get_type_and_id(dev_type)
        self.dev_type = dev_type
        self.dev_id = dev_id
        self.xp = _CPU_BACKEND if dev_type == "cpu" else _GPU_BACKEND

    def __eq__(self, other: Any) -> bool:
        return (
                isinstance(other, Device)
                and other.dev_type == self.dev_type
                and other.dev_id == self.dev_id
        )

    def __repr__(self) -> str:
        id_suffix = f":{self.dev_id}" if self.dev_type == "cuda" else ""
        return f"device('{self.dev_type}{id_suffix}')"

    def __str__(self) -> str:
        id_suffix = f":{self.dev_id}" if self.dev_type == "cuda" else ""
        return f"{self.dev_type}{id_suffix}"

    def __enter__(self) -> None:
        if self.dev_type == "cpu":
            return None
        return _GPU_BACKEND.cuda.Device(self.dev_id).__enter__()

    def __exit__(self, *args: Any) -> None:
        if self.dev_type == "cpu":
            return None
        return _GPU_BACKEND.cuda.Device(self.dev_id).__exit__(*args)


DeviceLike: TypeAlias = Device | str


def get_available_devices() -> list[str]:
    """Returns a list of available devices, including CPU and CUDA GPUs.

    Returns:
        list[str]: A list of device names (e.g., ["cpu", "cuda:0", "cuda:1", ...]).
    """
    devices = ["cpu"]
    if _GPU_BACKEND is not None:
        num_gpu_devices = _GPU_BACKEND.cuda.runtime.getDeviceCount()
        gpu_devices = [f"cuda:{i}" for i in range(num_gpu_devices)]
        devices.extend(gpu_devices)
    return devices


def get_array_device(x: ArrayLike) -> Device:
    """Determines the device of the given array.

    Args:
        x (ArrayLike): The input array.

    Returns:
        Device: A Device instance representing either CPU or CUDA.
    """
    return Device("cpu") if "numpy" in str(type(x)) else Device("cuda:0")


def select_device(device: Optional[DeviceLike]) -> Device:
    """Selects a device, defaulting to CPU if none is provided.

    Args:
        device (DeviceLike | None): The device to select.

    Returns:
        Device: A Device instance corresponding to the selected device.
    """
    if isinstance(device, Device):
        return device
    return Device(device or "cpu")


def parse_device(device: DeviceLike) -> Device:
    """Parses a device-like input into a Device instance.

    Args:
        device (DeviceLike): The device to parse.

    Returns:
        Device: A parsed Device instance.
    """
    return device if isinstance(device, Device) else Device(device)


def move_to_device(data: ArrayLike, device: Device) -> ArrayLike:
    """Moves an array to the specified device.

    Args:
        data (ArrayLike): The array to move.
        device (Device): The target device.

    Returns:
        ArrayLike: The array moved to the specified device.
    """
    if device == Device("cpu"):
        return _GPU_BACKEND.asnumpy(data)
    assert gpu_available(), "GPUs are not available."
    return cupy.asarray(data)
