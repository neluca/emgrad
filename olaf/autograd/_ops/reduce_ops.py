from typing import Optional
from olaf.operators import Op
from olaf.dtypes import ArrayLike


class Sum(Op):
    def forward(self, x: ArrayLike, *, dim: Optional[int | tuple[int, ...]], keepdims: bool) -> ArrayLike:
        y = x.sum(dim, keepdims=keepdims)
        self.save_to_cache(x.shape, dim, keepdims)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x_shape, dim, keepdims = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = self.xp.expand_dims(dy, dim)
        dx = self.xp.broadcast_to(dy, x_shape)
        return tuple((dx,))


class Mean(Op):
    def forward(self, x: ArrayLike, *, dim: Optional[int | tuple[int, ...]], keepdims: bool) -> ArrayLike:
        y = x.mean(dim, keepdims=keepdims)
        self.save_to_cache(x.shape, dim, keepdims, x.size / y.size)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x_shape, dim, keepdims, size = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = self.xp.expand_dims(dy, dim)
        dx = self.xp.broadcast_to(dy / size, x_shape)
        return tuple((dx,))


class Var(Op):
    def forward(
            self,
            x: ArrayLike,
            *,
            dim: Optional[int | tuple[int, ...]],
            ddof: int,
            keepdims: bool,
    ) -> ArrayLike:
        y = x.var(dim, ddof=ddof, keepdims=keepdims)
        self.save_to_cache(x, dim, x.size / y.size - ddof)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, dim, n = self.retrieve_from_cache()
        dx = dy * 2.0 * (x - x.mean(dim, keepdims=True)) / n
        return tuple((dx,))


class Std(Op):
    def forward(
            self,
            x: ArrayLike,
            *,
            dim: Optional[int | tuple[int, ...]],
            ddof: int,
            keepdims: bool,
    ) -> ArrayLike:
        y = x.std(dim, ddof=ddof, keepdims=keepdims)
        self.save_to_cache(x, dim, ddof, y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, dim, ddof, y = self.retrieve_from_cache()
        n = x.size / y.size - ddof
        dx = dy * (x - x.mean(dim, keepdims=True)) / (n * y)
        return tuple((dx,))


class Max(Op):
    def forward(self, x: ArrayLike, *, dim: Optional[int], keepdims: bool) -> ArrayLike:
        y = x.max(dim, keepdims=True)
        self.save_to_cache(dim, keepdims, x == y)
        return y if keepdims else y.squeeze()

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, keepdims, mask = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = self.xp.expand_dims(dy, dim)
        dx = mask * dy / mask.sum(dim, dtype=dy.dtype, keepdims=True)
        return tuple((dx,))


class Min(Op):
    def forward(self, x: ArrayLike, *, dim: Optional[int], keepdims: bool) -> ArrayLike:
        y = x.min(dim, keepdims=True)
        self.save_to_cache(dim, keepdims, x == y)
        return y if keepdims else y.squeeze()

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, keepdims, mask = self.retrieve_from_cache()
        if not keepdims and dim is not None:
            dy = self.xp.expand_dims(dy, dim)
        dx = mask * dy / mask.sum(dim, dtype=dy.dtype, keepdims=True)
        return tuple((dx,))
