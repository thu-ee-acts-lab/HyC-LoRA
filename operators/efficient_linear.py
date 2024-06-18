import torch
import bitsandbytes.functional as F
from .compress_function import (
    true_divide_outlier_suboutlinear_svd_compress,
    true_divide_outlier_suboutlinear_svd_decompress,
    get_statistics,
    pad_cut_L
)


class EfficientMemoryLinearFunc(torch.autograd.Function):
    # only suitable for batched matmul: (BxMxK) @ (KxR) -> (BxKxR) or (BxKxR) @ (RxN) -> (BxKxN)
    # and LoRA do not have bias
    @staticmethod
    def forward(
        ctx,
        x,
        w,
        b,
        use_bias,
        outlier_ratio,
        sub_outlier_ratio,
        sub_outlier_bit,
        sub_outlier_quantize_method,
        rank,
        iteration,
        static_value,
    ):
        ctx.use_bias = use_bias
        if use_bias:
            output = (
                x @ w.transpose(0, 1) + b[None, ...]
            )  # TODO: what is the dimension of b?
        else:
            output = x @ w.transpose(0, 1)
        
        # we just need to use the first batch to calculate the outlier
        if iteration < 2:
            outlier, L, R, scale = get_statistics(x, iteration, outlier_ratio, sub_outlier_ratio, sub_outlier_bit, sub_outlier_quantize_method, rank)
        else:
            outlier = static_value[0]
            L = static_value[1]
            scale = static_value[2]
            R = static_value[3]
            
        x_outlier_compressed, x_sub_outlier_compressed, scale = true_divide_outlier_suboutlinear_svd_compress(x, outlier, scale, sub_outlier_bit, sub_outlier_ratio, L, R)
        
        ctx.mark_non_differentiable(outlier, L, R, scale)
        ctx.x_outlier_compressed = x_outlier_compressed
        ctx.save_for_backward(x_sub_outlier_compressed, scale, w, L, R)
        ctx.sub_outlier_bit = sub_outlier_bit
        
        return output, outlier, L, R, scale

    @staticmethod
    def backward(ctx, grad_output, grad_outlier, grad_L, grad_R, grad_scale):
        
        use_bias = ctx.use_bias
        x_outlier_compressed = ctx.x_outlier_compressed
        x_sub_outlier_compressed, scale, w, L, R = ctx.saved_tensors
        x = true_divide_outlier_suboutlinear_svd_decompress(x_outlier_compressed, x_sub_outlier_compressed, ctx.sub_outlier_bit, scale, L=L, R=R)

        grad_input = grad_weight = grad_bias = None
        grad_output = grad_output.to(w.dtype)
        grad_input = grad_output @ w
        grad_output = grad_output.to(x.dtype)
        grad_weight = grad_output.transpose(-2, -1) @ x
        if use_bias:
            grad_bias = grad_output.sum(0)

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )


class EfficientMemoryLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        outlier_ratio: float = 0.01,
        sub_outlier_ratio: float = 0.2, #! initialize
        sub_outlier_bit: int = 8,
        sub_outlier_quantize_method: str = 'per-tensor',
        rank: int = 16,
    ):
        super(EfficientMemoryLinear, self).__init__(in_features, out_features, bias)
        self.outlier_ratio = outlier_ratio
        self.sub_outlier_ratio = sub_outlier_ratio
        self.sub_outlier_bit = sub_outlier_bit
        self.sub_outlier_quantize_method = sub_outlier_quantize_method
        self.rank = rank
        self.iteration = 0
        self.static_value = [None, None, None, None]

    def forward(self, input: torch.Tensor):
        result, outlier, L, R, scale = EfficientMemoryLinearFunc.apply(
            input,
            self.weight,
            self.bias,
            self.bias != None,
            self.outlier_ratio,
            self.sub_outlier_ratio,
            self.sub_outlier_bit,
            self.sub_outlier_quantize_method,
            self.rank,
            self.iteration,
            self.static_value,
        )

        if self.iteration < 2:
            self.static_value[0] = (
                outlier
                if self.static_value[0] is None
                else (self.iteration * self.static_value[0] + outlier)
                / (self.iteration + 1)
            )
            self.static_value[1] = (
                L
                if self.static_value[1] is None
                else (self.iteration * self.static_value[1] + pad_cut_L(L, self.static_value[1])) 
                / (self.iteration + 1)
            )
            self.static_value[2] = (
                scale
                if self.static_value[2] is None
                else (self.iteration * self.static_value[2] + scale)
                / (self.iteration + 1)
            )
            self.static_value[3] = (
                R
                if self.static_value[3] is None
                else (self.iteration * self.static_value[3] + R) 
                / (self.iteration + 1)
            )
        self.iteration += 1

        return result