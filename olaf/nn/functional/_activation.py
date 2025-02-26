from .. import _ops as NNOps
from olaf.autograd import Tensor, apply_op


def sigmoid(x: Tensor) -> Tensor:
    return apply_op(NNOps.Sigmoid, x)


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def relu(x: Tensor) -> Tensor:
    return apply_op(NNOps.ReLU, x)


def leaky_relu(x: Tensor, alpha: float = 0.2) -> Tensor:
    return apply_op(NNOps.LeakyReLU, x, alpha=alpha)


def gelu(x: Tensor) -> Tensor:
    return apply_op(NNOps.GELU, x)


def softmax(x: Tensor, *, dim: int = -1) -> Tensor:
    return apply_op(NNOps.Softmax, x, dim=dim)
