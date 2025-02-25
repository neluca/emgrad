from .op import Op
from olaf.dtypes import ArrayLike


class Add(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 + x2
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dx1 = dy
        dx2 = dy
        return tuple((dx1, dx2))


class Sub(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 - x2
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dx1 = dy
        dx2 = -dy
        return tuple((dx1, dx2))


class Mul(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 * x2
        self.save_to_cache(x1, x2)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = dy * x2
        dx2 = dy * x1
        return tuple((dx1, dx2))


class Div(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 / x2
        self.save_to_cache(x1, x2)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = dy / x2
        dx2 = -(dy * x1) / (x2 * x2)
        return tuple((dx1, dx2))


class Dot(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = x1 @ x2
        self.save_to_cache(x1, x2)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x1, x2 = self.retrieve_from_cache()
        dx1 = dy @ x2.swapaxes(-1, -2)  # dy @ x2.T
        dx2 = x1.swapaxes(-1, -2) @ dy  # x1.T @ dy
        return tuple((dx1, dx2))


class Maximum(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = self.xp.maximum(x1, x2)
        self.save_to_cache(y == x1)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        mask = self.retrieve_from_cache()
        dx1 = dy * mask
        dx2 = dy * self.xp.invert(mask)
        return tuple((dx1, dx2))


class Minimum(Op):
    def forward(self, x1: ArrayLike, x2: ArrayLike) -> ArrayLike:
        y = self.xp.minimum(x1, x2)
        self.save_to_cache(y == x1)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        mask = self.retrieve_from_cache()
        dx1 = dy * mask
        dx2 = dy * self.xp.invert(mask)
        return tuple((dx1, dx2))
