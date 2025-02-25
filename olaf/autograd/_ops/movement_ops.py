from itertools import accumulate
from typing import Any
from .op import Op
from olaf.dtypes import ArrayLike, ShapeLike


class Concat(Op):
    def forward(self, *arrays: ArrayLike, dim: int) -> ArrayLike:
        y = self.xp.concatenate(arrays, dim)
        self.save_to_cache(dim, [a.shape[dim] for a in arrays])
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, split_sizes = self.retrieve_from_cache()
        split_indices = list(accumulate(s for s in split_sizes))
        dxs = self.xp.split(dy, split_indices, dim)
        return tuple(dxs)


class Expand(Op):
    def forward(self, x: ArrayLike, *, shape: ShapeLike) -> ArrayLike:
        y = self.xp.broadcast_to(x, shape)
        return y

    def backward(self, dy: ArrayLike) -> ArrayLike:
        return dy


class Transpose(Op):
    def forward(self, x: ArrayLike, *, dim1: int, dim2: int) -> ArrayLike:
        y = x.swapaxes(dim1, dim2)
        self.save_to_cache(dim1, dim2)
        return y

    def backward(self, dy: ArrayLike) -> ArrayLike:
        dim1, dim2 = self.retrieve_from_cache()
        dx = dy.swapaxes(dim1, dim2)
        return dx


class Select(Op):
    def forward(self, x: ArrayLike, *, key: Any) -> ArrayLike:
        y = x[key]
        self.save_to_cache(x.shape, key)
        return y

    def backward(self, dy: ArrayLike) -> ArrayLike:
        x_shape, key = self.retrieve_from_cache()
        dx = self.xp.zeros(x_shape, dtype=dy.dtype)
        self.xp.add.at(dx, key, dy)
        return dx


class Split(Select):
    pass


class Stack(Op):
    def forward(self, *arrays: ArrayLike | bool, dim: int) -> ArrayLike:
        y = self.xp.stack(arrays, dim)
        self.save_to_cache(dim)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim = self.retrieve_from_cache()
        dxs = tuple(self.xp.moveaxis(dy, dim, 0))
        return tuple(dxs)


class View(Op):
    def forward(self, x: ArrayLike, *, shape: ShapeLike) -> ArrayLike:
        y = self.xp.reshape(x, shape)
        self.save_to_cache(x.shape)
        return y

    def backward(self, dy: ArrayLike) -> ArrayLike:
        (x_shape,) = self.retrieve_from_cache()
        dx = dy.reshape(x_shape)
        return dx


class Squeeze(View):
    pass


class Where(Op):
    def forward(
            self,
            condition: ArrayLike,
            x1: ArrayLike,
            x2: ArrayLike,
    ) -> ArrayLike:
        y = self.xp.where(condition, x1, x2)
        self.save_to_cache(y == x1)
        return y

    def backward(self, dy: ArrayLike) -> tuple[None, ArrayLike, ...]:
        mask = self.retrieve_from_cache()
        dx1 = dy * mask
        dx2 = dy * self.xp.invert(mask)
        return None, dx1, dx2
