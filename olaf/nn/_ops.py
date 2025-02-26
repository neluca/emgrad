from typing import Optional, Any
from types import ModuleType
from ..dtypes import ArrayLike, ShapeLike
from ..operators import Op


def _sigmoid_fwd(xp: ModuleType, x: ArrayLike) -> ArrayLike:
    return 1 / (1 + xp.exp(-x))


class Sigmoid(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = _sigmoid_fwd(self.xp, x)
        self.save_to_cache(y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (y,) = self.retrieve_from_cache()
        dx = dy * y * (1 - y)
        return tuple((dx,))


class ReLU(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = self.xp.maximum(x, 0.0)
        self.save_to_cache(y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (mask,) = self.retrieve_from_cache()
        dx = dy * mask
        return tuple((dx,))


class LeakyReLU(Op):
    def forward(self, x: ArrayLike, *, alpha: float) -> ArrayLike:
        y = self.xp.maximum(x, x * alpha)
        self.save_to_cache(alpha, y == x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        alpha, mask = self.retrieve_from_cache()
        dx = dy * (mask + self.xp.invert(mask).astype(dy.dtype) * alpha)
        return tuple((dx,))


class GELU(Op):
    def forward(self, x: ArrayLike) -> ArrayLike:
        y = 0.5 * x * (1 + self.xp.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
        self.save_to_cache(x)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (x,) = self.retrieve_from_cache()
        t = self.xp.tanh(x * 0.79788 * (1 + 0.04472 * x * x))
        dx = dy * 0.5 * ((1 + t) + x * (1 - t * t) * (0.79788 + 0.10703 * x * x))
        return tuple((dx,))


def _softmax_fwd(xp: ModuleType, x: ArrayLike, dim: int) -> ArrayLike:
    x = xp.exp(x - x.max(dim, keepdims=True))
    return x / x.sum(dim, keepdims=True)


def _softmax_bwd(y: ArrayLike, dy: ArrayLike, dim: int) -> ArrayLike:
    return y * (dy - (dy * y).sum(dim, keepdims=True))


class Softmax(Op):
    def forward(self, x: ArrayLike, *, dim: int) -> ArrayLike:
        y = _softmax_fwd(self.xp, x, dim)
        self.save_to_cache(dim, y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        dim, y = self.retrieve_from_cache()
        dx = _softmax_bwd(y, dy, dim)
        return tuple((dx,))


class Linear(Op):
    def forward(self, x: ArrayLike, w: ArrayLike, b: Optional[ArrayLike]) -> ArrayLike:
        y = x @ w.swapaxes(-1, -2)
        y = y if b is None else y + b
        self.save_to_cache(x, w, b is not None)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, b_requires_grad = self.retrieve_from_cache()
        dx = dy @ w
        dw = dy.swapaxes(-1, -2) @ x  # dy.T @ x
        db = None if not b_requires_grad else dy
        return tuple((dx, dw, db))


def _pad1d_fwd(
        xp: ModuleType, x: ArrayLike, left_pad: int, right_pad: Optional[int] = None
) -> ArrayLike:
    right_pad = right_pad if right_pad is not None else left_pad
    paddings = tuple([(0, 0)] * (x.ndim - 1) + [(left_pad, right_pad)])
    return xp.pad(x, paddings)


def _pad1d_bwd(
        dy: ArrayLike, left_pad: int, right_pad: Optional[int] = None
) -> ArrayLike:
    right_pad = right_pad if right_pad is not None else left_pad
    if right_pad <= 0:
        return dy[..., left_pad:]
    return dy[..., left_pad:-right_pad]


class Pad1D(Op):
    def forward(self, x: ArrayLike, *, padding: int) -> ArrayLike:
        y = _pad1d_fwd(self.xp, x, padding, padding)
        self.save_to_cache(padding)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (padding,) = self.retrieve_from_cache()
        dx = _pad1d_bwd(dy, padding, padding)
        return tuple((dx,))


def _dilate1d_fwd(xp: ModuleType, x: ArrayLike, dilation: int) -> ArrayLike:
    y_size = dilation * (x.shape[-1] - 1) + 1
    y_shape = (*x.shape[:-1], y_size)
    y = xp.zeros(y_shape, dtype=x.dtype)
    y[..., ::dilation] = x
    return y


def _dilate1d_bwd(dy: ArrayLike, dilation: int) -> ArrayLike:
    return dy[..., ::dilation]


class Dilate1D(Op):
    def forward(self, x: ArrayLike, *, dilation: int) -> ArrayLike:
        y = _dilate1d_fwd(self.xp, x, dilation)
        self.save_to_cache(dilation)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (dilation,) = self.retrieve_from_cache()
        dx = _dilate1d_bwd(dy, dilation)
        return tuple((dx,))


def _pad_to_shape(xp: ModuleType, x: ArrayLike, shape: ShapeLike) -> ArrayLike:
    padding = tuple((int(0), shape[i] - x.shape[i]) for i in range(x.ndim))
    return xp.pad(x, padding)


def _windowed_view_1d(
        xp: ModuleType, x: ArrayLike, window_size: int, stride: int = 1
) -> ArrayLike:
    # compute strided shape
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-1], out, window_size)

    # compute strides
    xstr = x.strides
    out_strides = (*xstr[:-1], xstr[-1] * stride, xstr[-1])

    return xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)


class Conv1D(Op):
    def forward(self, x: ArrayLike, w: ArrayLike, *, stride: int) -> ArrayLike:
        kernel_size = w.shape[-1]

        # convolve
        x_windowed = _windowed_view_1d(self.xp, x, kernel_size, stride)
        y = self.xp.einsum("bitk,oik->bot", x_windowed, w, use_blas=True)

        self.save_to_cache(x, w, x.shape[-1], kernel_size, stride)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, input_size, kernel_size, stride = self.retrieve_from_cache()

        # dilate and full pad
        if stride > 1:
            dy = _dilate1d_fwd(self.xp, dy, stride)
        output_size = input_size - kernel_size + 1
        output_shape = (*dy.shape[:-1], output_size)
        dy = _pad_to_shape(self.xp, dy, output_shape)
        dy = _pad1d_fwd(self.xp, dy, kernel_size - 1)

        # input grads
        w = self.xp.flip(w, axis=-1)
        dy_windowed = _windowed_view_1d(self.xp, dy, kernel_size)
        dx = self.xp.einsum("botk,oik->bit", dy_windowed, w, use_blas=True)

        # weight grads
        dy_windowed = _windowed_view_1d(self.xp, dy, input_size)
        dw = self.xp.einsum("bokt,bit->oik", dy_windowed, x, use_blas=True)
        dw = self.xp.flip(dw, axis=-1)

        return tuple((dx, dw))


def _pad2d_fwd(
        xp: ModuleType, x: ArrayLike, left_pad: int, right_pad: Optional[int] = None
) -> ArrayLike:
    right_pad = right_pad if right_pad is not None else left_pad
    paddings = tuple([(0, 0)] * (x.ndim - 2) + [(left_pad, right_pad)] * 2)
    return xp.pad(x, paddings)


def _pad2d_bwd(
        dy: ArrayLike, left_pad: int, right_pad: Optional[int] = None
) -> ArrayLike:
    right_pad = right_pad if right_pad is not None else left_pad
    if right_pad <= 0:
        return dy[..., left_pad:, left_pad:]
    return dy[..., left_pad:-right_pad, left_pad:-right_pad]


class Pad2D(Op):
    def forward(self, x: ArrayLike, *, padding: int) -> ArrayLike:
        y = _pad2d_fwd(self.xp, x, padding, padding)
        self.save_to_cache(padding)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (padding,) = self.retrieve_from_cache()
        dx = _pad2d_bwd(dy, padding, padding)
        return tuple((dx,))


class OutPad2D(Op):
    def forward(self, x: ArrayLike, *, padding: int, output_padding: int) -> ArrayLike:
        y = _pad2d_bwd(x, padding, padding - output_padding)
        self.save_to_cache(padding, output_padding)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        padding, output_padding = self.retrieve_from_cache()
        dx = _pad2d_fwd(self.xp, dy, padding, padding - output_padding)
        return tuple((dx,))


def _dilate2d_fwd(xp: ModuleType, x: ArrayLike, dilation: int) -> ArrayLike:
    y_size = dilation * (x.shape[-1] - 1) + 1
    y_shape = (*x.shape[:-2], y_size, y_size)
    y = xp.zeros(y_shape, dtype=x.dtype)
    y[..., ::dilation, ::dilation] = x
    return y


def _dilate2d_bwd(dy: ArrayLike, dilation: int) -> ArrayLike:
    return dy[..., ::dilation, ::dilation]


class Dilate2D(Op):
    def forward(self, x: ArrayLike, *, dilation: int) -> ArrayLike:
        y = _dilate2d_fwd(self.xp, x, dilation)
        self.save_to_cache(dilation)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        (dilation,) = self.retrieve_from_cache()
        dx = _dilate2d_bwd(dy, dilation)
        return tuple((dx,))


def _windowed_view_2d(
        xp: ModuleType, x: ArrayLike, window_size: int, stride: int = 1
) -> ArrayLike:
    # compute strided shape
    out = (x.shape[-1] - window_size) // stride + 1
    out_shape = (*x.shape[:-2], out, out, window_size, window_size)

    # compute strides
    xstr = x.strides
    out_strides = (*xstr[:-2], xstr[-2] * stride, xstr[-1] * stride, *xstr[-2:])

    return xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)


class Conv2D(Op):
    def forward(self, x: ArrayLike, w: ArrayLike, *, stride: int) -> ArrayLike:
        kernel_size = w.shape[-1]

        # convolve
        x_windowed = _windowed_view_2d(self.xp, x, kernel_size, stride)
        y = self.xp.einsum("biyxjk,oijk->boyx", x_windowed, w, use_blas=True)

        self.save_to_cache(x, w, x.shape[-1], kernel_size, stride)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, input_size, kernel_size, stride = self.retrieve_from_cache()

        # dilate and full pad
        if stride > 1:
            dy = _dilate2d_fwd(self.xp, dy, stride)
        output_size = input_size - kernel_size + 1
        output_shape = (*dy.shape[:-2], output_size, output_size)
        dy = _pad_to_shape(self.xp, dy, output_shape)
        dy = _pad2d_fwd(self.xp, dy, kernel_size - 1)

        # input grads
        w = self.xp.flip(w, axis=(-2, -1))
        dy_windowed = _windowed_view_2d(self.xp, dy, kernel_size)
        dx = self.xp.einsum("boyxjk,oijk->biyx", dy_windowed, w, use_blas=True)

        # weight grads
        dy_windowed = _windowed_view_2d(self.xp, dy, input_size)
        dw = self.xp.einsum("bojkyx,biyx->oijk", dy_windowed, x, use_blas=True)
        dw = self.xp.flip(dw, axis=(-2, -1))

        return tuple((dx, dw))


class ConvTranspose2D(Op):
    def forward(self, x: ArrayLike, w: ArrayLike, *, stride: int) -> ArrayLike:
        kernel_size = w.shape[-1]

        # dilate and full pad
        if stride > 1:
            x = _dilate2d_fwd(self.xp, x, stride)
        x = _pad2d_fwd(self.xp, x, kernel_size - 1)

        # convolve
        w = self.xp.flip(w, axis=(-2, -1))
        x_pooled = _windowed_view_2d(self.xp, x, kernel_size, 1)
        y = self.xp.einsum("biyxjk,oijk->boyx", x_pooled, w, use_blas=True)

        self.save_to_cache(x, w, x.shape[-1], kernel_size, stride)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, w, input_size, kernel_size, stride = self.retrieve_from_cache()

        # full pad
        dy = _pad2d_fwd(self.xp, dy, kernel_size - 1)

        # input grads
        w = self.xp.flip(w, axis=(-2, -1))
        dy_pooled = _windowed_view_2d(self.xp, dy, kernel_size)
        dx = self.xp.einsum("boyxjk,oijk->biyx", dy_pooled, w, use_blas=True)
        dx = _pad2d_bwd(dx, kernel_size - 1)
        dx = _dilate2d_bwd(dx, stride)

        # weight grads
        dy_pooled = _windowed_view_2d(self.xp, dy, input_size)
        dw = self.xp.einsum("bojkyx,biyx->oijk", dy_pooled, x, use_blas=True)

        return tuple((dx, dw))


def _repeat2d(xp: ModuleType, x: ArrayLike, n_repeats: int, target_shape: ShapeLike):
    out_shape = (*x.shape[:-1], n_repeats, x.shape[-1], n_repeats)
    out_strides = (*x.strides[:-1], 0, x.strides[-1], 0)
    y = xp.lib.stride_tricks.as_strided(x, out_shape, out_strides)
    y = y.reshape((*y.shape[:-4], y.shape[-4] * n_repeats, y.shape[-2] * n_repeats))
    return y if y.shape == target_shape else _pad_to_shape(xp, y, target_shape)


class MaxPool2D(Op):
    def forward(self, x: ArrayLike, *, window_size: int) -> ArrayLike:
        y = _windowed_view_2d(self.xp, x, window_size, window_size).max((-2, -1))
        self.save_to_cache(x, window_size, y)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x, window_size, y = self.retrieve_from_cache()
        mask = _repeat2d(self.xp, y, window_size, x.shape) == x
        dy_upsampling = _repeat2d(self.xp, dy, window_size, x.shape)
        dx = dy_upsampling * mask
        return tuple((dx,))


def _dropout_mask(xp: ModuleType, shape: ShapeLike, p: float) -> ArrayLike:
    return xp.random.rand(*shape) > p


def _dropout_fwd(x: ArrayLike, dropout_mask: ArrayLike, p: float) -> ArrayLike:
    return x * dropout_mask / (1 - p)


def _dropout_bwd(dy: ArrayLike, dropout_mask: ArrayLike, p: float) -> ArrayLike:
    return dy * dropout_mask / (1 - p)


class Dropout(Op):
    def forward(self, x: ArrayLike, *, p: float) -> ArrayLike:
        dropout_mask = _dropout_mask(self.xp, x.shape, p)
        y = _dropout_fwd(x, dropout_mask, p)
        self.save_to_cache(p, dropout_mask)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        p, dropout_mask = self.retrieve_from_cache()
        dx = _dropout_bwd(dy, dropout_mask, p)
        return tuple((dx,))


class ScaledDotProductAttention(Op):
    def forward(
            self,
            q: ArrayLike,
            k: ArrayLike,
            v: ArrayLike,
            attn_mask: Optional[ArrayLike],
            *,
            dropout_p: float,
    ) -> ArrayLike:
        *_, seq_len, head_size = q.shape

        attn = q @ k.swapaxes(-1, -2) / self.xp.sqrt(head_size)  # q @ k.T / scale
        if attn_mask is not None:
            attn += attn_mask[:seq_len, :seq_len]
        attn_weights = _softmax_fwd(self.xp, attn, dim=-1)
        if dropout_p > 0:
            drop_mask = _dropout_mask(self.xp, attn.shape, dropout_p)
            attn_weights = _dropout_fwd(attn_weights, drop_mask, dropout_p)
        y = attn_weights @ v

        self.save_to_cache(q, k, v, attn_weights, dropout_p)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike, None]:
        q, k, v, attn_weights, dropout_p = self.retrieve_from_cache()
        head_size = q.shape[-1]

        # attention gradients
        dattn_weights = dy @ v.swapaxes(-1, -2)  # dy @ v.T
        if dropout_p > 0:
            drop_mask = attn_weights != 0
            dattn_weights = _dropout_bwd(dattn_weights, drop_mask, dropout_p)
        dattn = _softmax_bwd(attn_weights, dattn_weights, -1)
        dattn /= self.xp.sqrt(head_size)

        # query, key, value gradients
        dq = dattn @ k
        dk = dattn.swapaxes(-1, -2) @ q  # dattn_weights.T @ q
        dv = attn_weights.swapaxes(-1, -2) @ dy  # attn_weights.T @ dy

        return tuple((dq, dk, dv, None))


class BatchNorm(Op):
    def forward(
            self,
            x: ArrayLike,
            w: ArrayLike,
            b: ArrayLike,
            rmean: ArrayLike,
            rvar: ArrayLike,
            *,
            momentum: float,
            eps: float,
            training: bool,
    ) -> ArrayLike:
        b_dims = (0,) + tuple(d for d in range(x.ndim) if d > 1)
        ext_shape = (1,) * (x.ndim - 2)
        n = x.size / w.size

        if training:
            mean = x.mean(b_dims, keepdims=True)
            xshift = x - mean
            var = (xshift * xshift).mean(b_dims, keepdims=True)

            rmean *= 1 - momentum
            rmean += mean.squeeze() * momentum
            rvar *= 1 - momentum
            rvar += n / (n - 1) * var.squeeze() * momentum
        else:
            mean = rmean.reshape(*rmean.shape, *ext_shape)
            xshift = x - mean
            var = rvar.reshape(*rvar.shape, *ext_shape)

        rstd = 1 / self.xp.sqrt(var + eps)
        xnorm = xshift * rstd
        w = w.reshape(*w.shape, *ext_shape)
        b = b.reshape(*b.shape, *ext_shape)
        y = xnorm * w + b

        self.save_to_cache(w, b_dims, rstd, xnorm)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike, None, None]:
        w, b_dims, rstd, xnorm = self.retrieve_from_cache()
        db = dy.sum(b_dims)
        dw = (dy * xnorm).sum(b_dims)
        dxnorm = dy * w
        dx = rstd * (
                dxnorm
                - dxnorm.mean(b_dims, keepdims=True)
                - xnorm * (dxnorm * xnorm).mean(b_dims, keepdims=True)
        )

        return tuple((dx, dw, db, None, None))


class LayerNorm(Op):
    def forward(
            self, x: ArrayLike, w: ArrayLike, b: ArrayLike, *, eps: float
    ) -> ArrayLike:
        f_dims = tuple(range(x.ndim - w.ndim, x.ndim))

        mean = x.mean(f_dims, keepdims=True)
        xshift = x - mean
        var = (xshift * xshift).mean(f_dims, keepdims=True)
        rstd = 1 / self.xp.sqrt(var + eps)
        xnorm = xshift * rstd
        y = xnorm * w + b

        self.save_to_cache(w, f_dims, rstd, xnorm)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        w, f_dims, rstd, xnorm = self.retrieve_from_cache()
        b_dims = tuple(range(dy.ndim - w.ndim))
        db = dy.sum(b_dims)
        dw = (dy * xnorm).sum(b_dims)
        dxnorm = dy * w
        dx = rstd * (
                dxnorm
                - dxnorm.mean(f_dims, keepdims=True)
                - xnorm * (dxnorm * xnorm).mean(f_dims, keepdims=True)
        )

        return tuple((dx, dw, db))


class Embedding(Op):
    def forward(self, x: ArrayLike, *, key: Any) -> ArrayLike:
        y = x[key]
        self.save_to_cache(x.shape, key)
        return y

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, ...]:
        x_shape, key = self.retrieve_from_cache()
        dx = self.xp.zeros(x_shape, dtype=dy.dtype)
        self.xp.add.at(dx, key, dy)
        return tuple((dx,))


def _onehot(xp: ModuleType, x: ArrayLike, n: int, dtype: type):
    return xp.eye(n, dtype=dtype)[x]


class MSELoss(Op):
    def forward(self, x: ArrayLike, y: ArrayLike, *, reduction: str) -> ArrayLike:
        diff = x - y
        loss = diff * diff
        loss = loss.mean() if reduction == "mean" else loss.sum()
        self.save_to_cache(x.size, diff, reduction)
        return loss

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, None]:
        x_size, diff, reduction = self.retrieve_from_cache()
        dx = dy * 2.0 * diff
        if reduction == "mean":
            dx /= float(x_size)
        return tuple((dx, None))


class CrossEntropyLoss(Op):
    def forward(
            self, x: ArrayLike, y: ArrayLike, *, eta: float, reduction: str
    ) -> ArrayLike:
        probs = _softmax_fwd(self.xp, x, dim=-1)
        y_onehot = _onehot(self.xp, y, x.shape[-1], probs.dtype)
        loss = -(self.xp.log(probs + eta) * y_onehot).sum(-1)
        loss = loss.mean() if reduction == "mean" else loss.sum()
        self.save_to_cache(y, probs, reduction)
        return loss

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, None]:
        y, probs, reduction = self.retrieve_from_cache()
        y = _onehot(self.xp, y, probs.shape[-1], probs.dtype)
        dx = dy * (probs - y)
        if reduction == "mean":
            dx /= self.xp.prod(y.shape[:-1], dtype=dx.dtype)
        return tuple((dx, None))


class BCELoss(Op):
    def forward(self, x: ArrayLike, y: ArrayLike, *, reduction: str) -> ArrayLike:
        max_logits = self.xp.maximum(x, 0.0)
        loss = max_logits - x * y + self.xp.log(1 + self.xp.exp(-self.xp.abs(x)))
        loss = loss.mean() if reduction == "mean" else loss.sum()
        self.save_to_cache(x, y, reduction)
        return loss

    def backward(self, dy: ArrayLike) -> tuple[ArrayLike, None]:
        x, y, reduction = self.retrieve_from_cache()
        dx = dy * (_sigmoid_fwd(self.xp, x) - y)
        if reduction == "mean":
            dx /= float(x.size)
        return tuple((dx, None))
