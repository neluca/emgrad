from abc import ABC, abstractmethod
from typing import Any, Optional

from olaf._backends import Device
from olaf.dtypes import ArrayLike


class Op(ABC):
    """Base class for a differentiable operation.

    Args:
        device (Device): The device used for computations.
    """

    def __init__(self, device: Device, kwargs: Any) -> None:
        self.xp = device.xp
        self.kwargs = kwargs  # just for graph visualization
        self._cache: Any = None

    @property
    def name(self) -> str:
        """Returns the operation name."""
        return self.__class__.__name__

    def save_to_cache(self, *args: Any):
        """Saves values to the cache.

        Args:
            *args: Values to be cached.
        """
        self._cache = args

    def retrieve_from_cache(self) -> Any:
        """Retrieves the cached values and resets the cache afterwards.

        Returns:
            tuple[Any, ...]: The cached values.

        Raises:
            AssertionError: If no values are cached.
        """
        assert self._cache is not None
        values, self._cache = self._cache, None  # reset cache to None
        return values

    @abstractmethod
    def forward(self, *arrays: Optional[ArrayLike], **kwargs: Any) -> ArrayLike:
        """Computes the forward pass of the operation.

        Args:
            *arrays (ArrayLike | None): The input arrays.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ArrayLike: The result of the forward pass.
        """

    @abstractmethod
    def backward(self, dy: ArrayLike) -> ArrayLike | tuple[ArrayLike, ...] | tuple[None, ArrayLike, ...]:
        """Computes the backward pass (gradient) of the operation.

        Args:
            dy (ArrayLike): The gradient with respect to the output.

        Returns:
            tuple[ArrayLike, ...]: The gradient with respect to the inputs.
        """
