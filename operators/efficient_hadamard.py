import torch
from .compress_function import (
    true_divide_outlier_suboutlinear_svd_compress,
    true_divide_outlier_suboutlinear_svd_decompress,
    get_statistics,
    pad_cut_L
)

class EfficientMemoryHadamardFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x1,
        x2,
        outlier_ratio_1,
        sub_outlier_ratio_1,
        sub_outlier_bit_1,
        sub_outlier_quantize_method_1,
        outlier_ratio_2,
        sub_outlier_ratio_2,
        sub_outlier_bit_2,
        sub_outlier_quantize_method_2,
        rank, 
        iteration,
        static_value_1,
        static_value_2,
    ):
        result = x1 * x2
        
        # we just need to use the first batch to calculate the outlier
        # for the value 1
        if iteration < 5:
            outlier_1, L_1, R_1, scale_1 = get_statistics(x1, iteration, outlier_ratio_1, sub_outlier_ratio_1, sub_outlier_bit_1, sub_outlier_quantize_method_1, rank)
            outlier_2, L_2, R_2, scale_2 = get_statistics(x2, iteration, outlier_ratio_2, sub_outlier_ratio_2, sub_outlier_bit_2, sub_outlier_quantize_method_2, rank)
        else:
            outlier_1 = static_value_1[0]
            L_1 = static_value_1[1]
            scale_1 = static_value_1[2]
            R_1 = static_value_1[3]
            outlier_2 = static_value_2[0]
            L_2 = static_value_2[1]
            scale_2 = static_value_2[2]
            R_2 = static_value_2[3]
        
        x1_outlier_compressed, x1_sub_outlier_compressed, scale1 = true_divide_outlier_suboutlinear_svd_compress(x1, outlier_1, scale_1, sub_outlier_bit_1, sub_outlier_ratio_1, L_1, R_1)
        x2_outlier_compressed, x2_sub_outlier_compressed, scale2 = true_divide_outlier_suboutlinear_svd_compress(x2, outlier_2, scale_2, sub_outlier_bit_2, sub_outlier_ratio_2, L_2, R_2)
        
        ctx.sub_outlier_bit_1 = sub_outlier_bit_1
        ctx.sub_outlier_bit_2 = sub_outlier_bit_2
        ctx.mark_non_differentiable(outlier_1, outlier_2, L_1, R_1, scale_1, L_2, R_2, scale_2)
        ctx.save_for_backward(x1_outlier_compressed, x1_sub_outlier_compressed, scale1, L_1, R_1, x2_outlier_compressed, x2_sub_outlier_compressed, scale2, L_2, R_2)

        return result, outlier_1, L_1, R_1, scale_1, outlier_2, L_2, R_2, scale_2
            
    def backward(ctx, grad_output, grad_outlier_1, grad_L1, grad_R1, grad_scale_1, grad_outlier_2, grad_L2, grad_R2, grad_scale_2):
        x1_outlier_compressed, x1_sub_outlier_compressed, scale1, L_1, R_1, x2_outlier_compressed, x2_sub_outlier_compressed, scale2, L_2, R_2 = ctx.saved_tensors
        grad_input1, grad_input2 = None, None
        
        x1 = true_divide_outlier_suboutlinear_svd_decompress(x1_outlier_compressed, x1_sub_outlier_compressed, ctx.sub_outlier_bit_1, scale1, L=L_1, R=R_1)
        x2 = true_divide_outlier_suboutlinear_svd_decompress(x2_outlier_compressed, x2_sub_outlier_compressed, ctx.sub_outlier_bit_2, scale2, L=L_2, R=R_2)

        grad_input1 = grad_output * x2
        grad_input2 = x1 * grad_output

        return (
            grad_input1,
            grad_input2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )


class EfficientMemoryHadamard(torch.nn.Module):
    def __init__(
        self,
        outlier_ratio_1: float = 0.01,
        sub_outlier_ratio_1: float = 0.2,
        sub_outlier_bit_1: int = 8,
        sub_outlier_quantize_method_1: str = 'per-tensor',
        outlier_ratio_2: float = 0.01,
        sub_outlier_ratio_2: float = 0.2,
        sub_outlier_bit_2: int = 8,
        sub_outlier_quantize_method_2: str = 'per-tensor',
        rank: int = 16,
    ):
        super(EfficientMemoryHadamard, self).__init__()
        self.outlier_ratio_1 = outlier_ratio_1
        self.sub_outlier_ratio_1 = sub_outlier_ratio_1
        self.sub_outlier_bit_1 = sub_outlier_bit_1
        self.sub_outlier_quantize_method_1 = sub_outlier_quantize_method_1
        self.outlier_ratio_2 = outlier_ratio_2
        self.sub_outlier_ratio_2 = sub_outlier_ratio_2
        self.sub_outlier_bit_2 = sub_outlier_bit_2
        self.sub_outlier_quantize_method_2 = sub_outlier_quantize_method_2
        self.rank = rank
        self.iteration = 0
        self.static_value_1 = [None, None, None, None]
        self.static_value_2 = [None, None, None, None]

    def forward(self, x1, x2):
        result, outlier_1, L_1, R_1, scale_1, outlier_2, L_2, R_2, scale_2 = EfficientMemoryHadamardFunc.apply(
            x1,
            x2,
            self.outlier_ratio_1,
            self.sub_outlier_ratio_1,
            self.sub_outlier_bit_1,
            self.sub_outlier_quantize_method_1,
            self.outlier_ratio_2,
            self.sub_outlier_ratio_2,
            self.sub_outlier_bit_2,
            self.sub_outlier_quantize_method_2,
            self.rank,
            self.iteration,
            self.static_value_1,
            self.static_value_2,
        )
        if self.iteration < 5:
            self.static_value_1[0] = (
                outlier_1
                if self.static_value_1[0] is None
                else (self.iteration * self.static_value_1[0] + outlier_1)
                / (self.iteration + 1)
            )
            self.static_value_1[1] = (
                L_1
                if self.static_value_1[1] is None
                else (self.iteration * self.static_value_1[1] + pad_cut_L(L_1, self.static_value_1[1]))
                / (self.iteration + 1)
            )
            self.static_value_1[2] = (
                scale_1
                if self.static_value_1[2] is None
                else (self.iteration * self.static_value_1[2] + scale_1)
                / (self.iteration + 1)
            )
            self.static_value_1[3] = (
                R_1
                if self.static_value_1[3] is None
                else (self.iteration * self.static_value_1[3] + R_1)
                / (self.iteration + 1)
            )
            self.static_value_2[0] = (
                outlier_2
                if self.static_value_2[0] is None
                else (self.iteration * self.static_value_2[0] + outlier_2)
                / (self.iteration + 1)
            )
            self.static_value_2[1] = (
                L_2
                if self.static_value_2[1] is None
                else (self.iteration * self.static_value_2[1] + pad_cut_L(L_2, self.static_value_2[1]))
                / (self.iteration + 1)
            )
            self.static_value_2[2] = (
                scale_2
                if self.static_value_2[2] is None
                else (self.iteration * self.static_value_2[2] + scale_2)
                / (self.iteration + 1)
            )
            self.static_value_2[3] = (
                R_2
                if self.static_value_2[3] is None
                else (self.iteration * self.static_value_2[3] + R_2)
                / (self.iteration + 1)
            )
            
        self.iteration += 1
        torch.cuda.empty_cache()
        return result

