import torch
import torch.nn.functional as F
from .compress_function import (
    prune_softmax,
    true_compress_softmax,
    true_decompress_softmax,
    get_statistics_softmax
)

class EfficientMemorySoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        outlier_ratio,
        iteration,
        static_value,
    ):
        y_return = F.softmax(x, dim=-1)
        y = y_return.clone()
        
        if iteration < 5:
            outlier = get_statistics_softmax(y, iteration, outlier_ratio)
        else:
            outlier = static_value

        y_sparse = true_compress_softmax(y, outlier)
        
        ctx.mark_non_differentiable(outlier)
        ctx.save_for_backward(y_sparse)
        
        return y_return, outlier

    @staticmethod
    def backward(ctx, grad_output, grad_outlier):
        (y_sparse,)  = ctx.saved_tensors
        y = true_decompress_softmax(y_sparse)

        return (
            (grad_output - (grad_output * y).sum(dim=-1, keepdims=True)) * y,
            None,
            None,
            None,
        )


class EfficientMemorySoftmax(torch.nn.Module):
    def __init__(
        self,
        outlier_ratio: float = 0.01,
    ):
        super(EfficientMemorySoftmax, self).__init__()
        self.outlier_ratio = outlier_ratio
        self.iteration = 0
        self.static_value = None

    def forward(self, x):
        result, outlier = EfficientMemorySoftmaxFunc.apply(
            x,
            self.outlier_ratio,
            self.iteration,
            self.static_value,
        )
        
        if self.iteration < 5:
            self.static_value = (
                outlier
                if self.static_value is None
                else (self.iteration * self.static_value + outlier)
                / (self.iteration + 1)
            )
        self.iteration += 1

        return result
