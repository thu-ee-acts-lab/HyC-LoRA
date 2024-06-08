import torch
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward
from .compress_function import (
    true_divide_outlier_suboutlinear_svd_compress,
    true_divide_outlier_suboutlinear_svd_decompress,
    get_statistics,
    pad_cut_L
)

class EfficientFlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        # for calibration
        outlier_ratio,
        sub_outlier_ratio,
        sub_outlier_bit,
        sub_outlier_quantize_method,
        rank,
        iteration,
        q_static_value,
        k_static_value,
        v_static_value,
        o_static_value
    ):
        
        # qkv: [B, L, 3, NH, HD]
        num_heads = qkv.shape[-2]
        ctx.num_heads = num_heads
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=None,
        )
        # [8192, 3, 32, 128])   
        # q: original: [8192, 4096] -> [32, 8192, 128] now: [16384, 16, 128] / [8192, 16, 128]
        # so: [16384, 16, 128] -> [16, 16384, 128] -> [1, 16, 16384, 128]
        q = q.permute(1, 0, 2).unsqueeze(0)
        k = k.permute(1, 0, 2).unsqueeze(0)
        v = v.permute(1, 0, 2).unsqueeze(0)
        out_padded = out_padded.permute(1, 0, 2).unsqueeze(0)
        
        if iteration < 5:
            q_outlier, q_L, q_R, q_scale = get_statistics(q, iteration, outlier_ratio, sub_outlier_ratio, sub_outlier_bit, sub_outlier_quantize_method, rank)
            k_outlier, k_L, k_R, k_scale = get_statistics(k, iteration, outlier_ratio, sub_outlier_ratio, sub_outlier_bit, sub_outlier_quantize_method, rank)
            v_outlier, v_L, v_R, v_scale = get_statistics(v, iteration, outlier_ratio, sub_outlier_ratio, sub_outlier_bit, sub_outlier_quantize_method, rank)
            o_outlier, o_L, o_R, o_scale = get_statistics(out_padded, iteration, outlier_ratio, sub_outlier_ratio, sub_outlier_bit, sub_outlier_quantize_method, rank)
        else:
            q_outlier, q_L, q_scale, q_R = q_static_value
            k_outlier, k_L, k_scale, k_R = k_static_value
            v_outlier, v_L, v_scale, v_R = v_static_value
            o_outlier, o_L, o_scale, o_R = o_static_value
        
        q_outlier_compressed, q_sub_outlier_compressed, q_scale = true_divide_outlier_suboutlinear_svd_compress(q, q_outlier, q_scale, sub_outlier_bit, sub_outlier_ratio, L=q_L, R=q_R)
        k_outlier_compressed, k_sub_outlier_compressed, k_scale = true_divide_outlier_suboutlinear_svd_compress(k, k_outlier, k_scale, sub_outlier_bit, sub_outlier_ratio, L=k_L, R=k_R)
        v_outlier_compressed, v_sub_outlier_compressed, v_scale = true_divide_outlier_suboutlinear_svd_compress(v, v_outlier, v_scale, sub_outlier_bit, sub_outlier_ratio, L=v_L, R=v_R)
        o_outlier_compressed, o_sub_outlier_compressed, o_scale = true_divide_outlier_suboutlinear_svd_compress(out_padded, o_outlier, o_scale, sub_outlier_bit, sub_outlier_ratio, L=o_L, R=o_R)
        
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.sub_outlier_bit = sub_outlier_bit
        
        ctx.mark_non_differentiable(q_outlier, k_outlier, v_outlier, o_outlier, q_L, q_R, k_L, k_R, v_L, v_R, o_L, o_R, q_scale, k_scale, v_scale, o_scale)
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens, rng_state)
        ctx.save_for_backward(q_outlier_compressed, q_sub_outlier_compressed, q_scale, k_outlier_compressed, k_sub_outlier_compressed, k_scale, v_outlier_compressed, v_sub_outlier_compressed, v_scale, o_outlier_compressed, o_sub_outlier_compressed, o_scale, q_L, q_R, k_L, k_R, v_L, v_R, o_L, o_R, softmax_lse, cu_seqlens, rng_state)
        return out, q_outlier, k_outlier, v_outlier, o_outlier, q_scale, k_scale, v_scale, o_scale, q_L, q_R, k_L, k_R, v_L, v_R, o_L, o_R

    @staticmethod
    def backward(ctx, dout, *args):
        q_outlier_compressed, q_sub_outlier_compressed, q_scale, k_outlier_compressed, k_sub_outlier_compressed, k_scale, v_outlier_compressed, v_sub_outlier_compressed, v_scale, o_outlier_compressed, o_sub_outlier_compressed, o_scale, q_L, q_R, k_L, k_R, v_L, v_R, o_L, o_R, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        q = true_divide_outlier_suboutlinear_svd_decompress(q_outlier_compressed, q_sub_outlier_compressed, ctx.sub_outlier_bit, q_scale, True, ctx.num_heads, L=q_L, R=q_R)
        k = true_divide_outlier_suboutlinear_svd_decompress(k_outlier_compressed, k_sub_outlier_compressed, ctx.sub_outlier_bit, k_scale, True, ctx.num_heads, L=k_L, R=k_R)
        v = true_divide_outlier_suboutlinear_svd_decompress(v_outlier_compressed, v_sub_outlier_compressed, ctx.sub_outlier_bit, v_scale, True, ctx.num_heads, L=v_L, R=v_R)
        out = true_divide_outlier_suboutlinear_svd_decompress(o_outlier_compressed, o_sub_outlier_compressed, ctx.sub_outlier_bit, o_scale, True, ctx.num_heads, L=o_L, R=o_R)
        
        # [1, 16, 16384, 128] -> [16, 16384, 128] -> [16384, 16, 128]
        q = q.squeeze(0).permute(1, 0, 2)
        k = k.squeeze(0).permute(1, 0, 2)
        v = v.squeeze(0).permute(1, 0, 2)
        out = out.squeeze(0).permute(1, 0, 2)
        
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    
class EfficientFlashAttnVarlenQKVPacked(torch.nn.Module):
    def __init__(
        self,
        outlier_ratio: float = 0.01,
        sub_outlier_ratio: float = 0.2, #! initialize
        sub_outlier_bit: int = 8,
        sub_outlier_quantize_method: str = 'per-tensor',
        rank: int = 16,
    ):
        super(EfficientFlashAttnVarlenQKVPacked, self).__init__()
        self.outlier_ratio = outlier_ratio
        self.sub_outlier_ratio = sub_outlier_ratio
        self.sub_outlier_bit = sub_outlier_bit
        self.sub_outlier_quantize_method = sub_outlier_quantize_method
        self.rank = rank
        self.iteration = 0
        
        self.q_static_value = [None, None, None, None]
        self.k_static_value = [None, None, None, None]
        self.v_static_value = [None, None, None, None]
        self.o_static_value = [None, None, None, None]
        
    def forward(
        self,
        qkv, 
        cu_seqlens,
        max_seqlen,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ):
        out, q_outlier, k_outlier, v_outlier, o_outlier, q_scale, k_scale, v_scale, o_scale, q_L, q_R, k_L, k_R, v_L, v_R, o_L, o_R = EfficientFlashAttnVarlenQKVPackedFunc.apply(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            # for calibration
            self.outlier_ratio,
            self.sub_outlier_ratio,
            self.sub_outlier_bit,
            self.sub_outlier_quantize_method,
            self.rank,
            self.iteration,
            self.q_static_value,
            self.k_static_value,
            self.v_static_value,
            self.o_static_value
        )
        
        if self.iteration < 5:
            # outlier
            self.q_static_value[0] = (
                q_outlier
                if self.q_static_value[0] is None
                else (self.iteration * self.q_static_value[0] + q_outlier)
                / (self.iteration + 1)
            )
            self.k_static_value[0] = (
                k_outlier
                if self.k_static_value[0] is None
                else (self.iteration * self.k_static_value[0] + k_outlier)
                / (self.iteration + 1)
            )
            self.v_static_value[0] = (
                v_outlier
                if self.v_static_value[0] is None
                else (self.iteration * self.v_static_value[0] + v_outlier)
                / (self.iteration + 1)
            )
            self.o_static_value[0] = (
                o_outlier
                if self.o_static_value[0] is None
                else (self.iteration * self.o_static_value[0] + o_outlier)
                / (self.iteration + 1)
            )
            
            # scale
            self.q_static_value[2] = (
                q_scale
                if self.q_static_value[2] is None
                else (self.iteration * self.q_static_value[2] + q_scale)
                / (self.iteration + 1)
            )
            self.k_static_value[2] = (
                k_scale
                if self.k_static_value[2] is None
                else (self.iteration * self.k_static_value[2] + k_scale)
                / (self.iteration + 1)
            )
            self.v_static_value[2] = (
                v_scale
                if self.v_static_value[2] is None
                else (self.iteration * self.v_static_value[2] + v_scale)
                / (self.iteration + 1)
            )
            self.o_static_value[2] = (
                o_scale
                if self.o_static_value[2] is None
                else (self.iteration * self.o_static_value[2] + o_scale)
                / (self.iteration + 1)
            )
            
            # L
            self.q_static_value[1] = (
                q_L
                if self.q_static_value[1] is None
                else (self.iteration * self.q_static_value[1] + pad_cut_L(q_L, self.q_static_value[1]))
                / (self.iteration + 1)
            )
            self.k_static_value[1] = (
                k_L
                if self.k_static_value[1] is None
                else (self.iteration * self.k_static_value[1] + pad_cut_L(k_L, self.k_static_value[1]))
                / (self.iteration + 1)
            )
            self.v_static_value[1] = (
                v_L
                if self.v_static_value[1] is None
                else (self.iteration * self.v_static_value[1] + pad_cut_L(v_L, self.v_static_value[1]))
                / (self.iteration + 1)
            )
            self.o_static_value[1] = (
                o_L
                if self.o_static_value[1] is None
                else (self.iteration * self.o_static_value[1] + pad_cut_L(o_L, self.o_static_value[1]))
                / (self.iteration + 1)
            )
            
            # R
            self.q_static_value[3] = (
                q_R
                if self.q_static_value[3] is None
                else (self.iteration * self.q_static_value[3] + q_R)
                / (self.iteration + 1)
            )
            self.k_static_value[3] = (
                k_R
                if self.k_static_value[3] is None
                else (self.iteration * self.k_static_value[3] + k_R)
                / (self.iteration + 1)
            )
            self.v_static_value[3] = (
                v_R
                if self.v_static_value[3] is None
                else (self.iteration * self.v_static_value[3] + v_R)
                / (self.iteration + 1)
            )
            self.o_static_value[3] = (
                o_R
                if self.o_static_value[3] is None
                else (self.iteration * self.o_static_value[3] + o_R)
                / (self.iteration + 1)
            )
        self.iteration += 1
        
        return out