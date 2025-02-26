from typing import Optional
from .. import _ops as NNOps
from olaf.autograd import Tensor, apply_op


def conv1d(
        x: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
) -> Tensor:
    if padding > 0:
        x = apply_op(NNOps.Pad1D, x, padding=padding)
    if dilation > 1:
        w = apply_op(NNOps.Dilate1D, w, dilation=dilation)
    y = apply_op(NNOps.Conv1D, x, w, stride=stride)
    if b is not None:
        y += b.view(*b.shape, 1)
    return y


def conv2d(
        x: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
) -> Tensor:
    if padding > 0:
        x = apply_op(NNOps.Pad2D, x, padding=padding)
    if dilation > 1:
        w = apply_op(NNOps.Dilate2D, w, dilation=dilation)
    y = apply_op(NNOps.Conv2D, x, w, stride=stride)
    if b is not None:
        y += b.view(*b.shape, 1, 1)
    return y


def conv_transpose2d(
        x: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
) -> Tensor:
    assert output_padding <= padding, "Output padding must be <= padding."
    if dilation > 1:
        w = apply_op(NNOps.Dilate2D, w, dilation=dilation)
    y = apply_op(NNOps.ConvTranspose2D, x, w, stride=stride)
    if padding > 0:
        y = apply_op(NNOps.OutPad2D, y, padding=padding, output_padding=output_padding)
    if b is not None:
        y += b.view(*b.shape, 1, 1)
    return y
