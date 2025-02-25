from itertools import accumulate
from .op import Op
from olaf.dtypes import ArrayLike, ShapeLike


class Concat(Op):
    """Concatenates arrays."""

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
    """Broadcasts array elements."""

    def forward(self, x: ArrayLike, *, shape: ShapeLike) -> ArrayLike:
        y = self.xp.broadcast_to(x, shape)
        return y

    def backward(self, dy: ArrayLike) -> ArrayLike:
        return dy
