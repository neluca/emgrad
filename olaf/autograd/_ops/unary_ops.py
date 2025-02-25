from .op import Op
from olaf.dtypes import ArrayLike, Scalar


class Abs(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = self.xp.absolute(x)
        self.save_to_cache(y != x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (mask,) = self.retrieve_from_cache()
        dx = dy
        self.xp.multiply.at(dx, mask, -1)
        return tuple((dx,))


class Exp(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = self.xp.exp(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * y
        return tuple((dx,))


class Log(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = self.xp.log(x)
        self.save_to_cache(x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (x,) = self.retrieve_from_cache()
        dx = dy / x
        return tuple((dx,))


class Pow(Op):
    def forward(self, x: ArrayLike, *, exp: Scalar) -> ArrayLike:
        y = x ** exp
        self.save_to_cache(x, exp)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, exp = self.retrieve_from_cache()
        dx = dy * exp * x ** (exp - 1)
        return tuple((dx,))


class Sqrt(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = self.xp.sqrt(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * 0.5 / y
        return tuple((dx,))


class Tanh(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = self.xp.tanh(x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * (1 - y * y)
        return tuple((dx,))


class Tril(Op):
    def forward(self, x: ArrayLike, *, diag: int) -> ArrayLike:
        y = self.xp.tril(x, diag)
        self.save_to_cache(y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (mask,) = self.retrieve_from_cache()
        dx = dy * mask
        return tuple((dx,))


class Triu(Op):
    def forward(self, x: ArrayLike, *, diag: int) -> ArrayLike:
        y = self.xp.triu(x, diag)
        self.save_to_cache(y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (mask,) = self.retrieve_from_cache()
        dx = dy * mask
        return tuple((dx,))
