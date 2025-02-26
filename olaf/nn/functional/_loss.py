from typing import Literal
from .. import _ops as NNOps
from olaf.autograd import Tensor, apply_op
from olaf.dtypes import is_int


def mse_loss(
        logits: Tensor, targets: Tensor, reduction: Literal["sum", "mean"] = "mean"
) -> Tensor:
    assert not targets.req_grad, "Targets cannot require gradients."
    return apply_op(NNOps.MSELoss, logits, targets, reduction=reduction)


def cross_entropy_loss(
        logits: Tensor,
        targets: Tensor,
        eta: float = 1e-8,
        reduction: Literal["sum", "mean"] = "mean",
) -> Tensor:
    assert not targets.req_grad, "Targets cannot require gradients."
    assert is_int(targets.dtype), "Targets must be integers."
    return apply_op(
        NNOps.CrossEntropyLoss, logits, targets, eta=eta, reduction=reduction
    )


def bce_loss(
        logits: Tensor, targets: Tensor, reduction: Literal["sum", "mean"] = "mean"
) -> Tensor:
    assert not targets.req_grad, "Targets cannot require gradients."
    return apply_op(NNOps.BCELoss, logits, targets, reduction=reduction)
