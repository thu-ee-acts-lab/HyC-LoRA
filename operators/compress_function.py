import torch
from .fused_compression import low_rank_extraction, low_rank_addition

def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)


def head_to_hidden_shape(x: torch.Tensor):
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)


def fake_svd_lowrank_simple_tensor(input: torch.Tensor, q: int, niter: int = 1):
    batch, seq_len, model_dim = input.shape
    input = input.reshape(batch * seq_len, model_dim)
    U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
    V = V.transpose(-1, -2)
    S = torch.diag_embed(S)
    output = torch.matmul(U[..., :, :], S[..., :, :])
    output = torch.matmul(output[..., :, :], V[..., :, :])
    output = output.reshape(batch, seq_len, model_dim)
    return output


def fake_svd_lowrank_head(input: torch.Tensor, q: int, niter: int = 2):
    batch, num_head, seq_len, sep_dim = input.shape
    input = input.permute(0, 2, 1, 3).reshape(batch * seq_len, num_head * sep_dim)
    U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
    V = V.transpose(-1, -2)
    S = torch.diag_embed(S)
    output = torch.matmul(U[..., :, :], S[..., :, :])
    output = torch.matmul(output[..., :, :], V[..., :, :])
    output = output.reshape(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output


def svd_lowrank_simple_tensor_compress(input: torch.Tensor, q: int, niter: int = 2):
    batch, seq_len, model_dim = input.shape
    input = input.reshape(batch * seq_len, model_dim)
    U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
    V = V.transpose(-1, -2)
    return U, S, V


def svd_lowrank_simple_tensor_decompress(U: torch.tensor, S: torch.tensor, V: torch.tensor, input_shape: torch.Size):
    batch, seq_len, model_dim = input_shape
    output = torch.matmul(U[..., :, :], S[..., :, :])
    output = torch.matmul(output[..., :, :], V[..., :, :])
    output = output.reshape(batch, seq_len, model_dim)
    return output


def svd_lowrank_head_compress(input: torch.Tensor, q: int, niter: int = 2):
    batch, num_head, seq_len, sep_dim = input.shape
    input = input.permute(0, 2, 1, 3).reshape(batch * seq_len, num_head * sep_dim)
    U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
    V = V.transpose(-1, -2)
    return U, S, V


def svd_lowrank_head_decompress(U: torch.tensor, S: torch.tensor, V: torch.tensor, input_shape: torch.Size):
    batch, seq_len, num_head, sep_dim = input_shape
    output = torch.matmul(U[..., :, :], S[..., :, :])
    output = torch.matmul(output[..., :, :], V[..., :, :])
    output = output.reshape(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output


def convert_coo_to_tuple(x_coo):
    return x_coo
    # extract all the elements
    x_coo_indices = x_coo.indices().to(torch.int16)
    x_coo_values = x_coo.values()
    x_coo_size = x_coo.size()
    x_coo_device = x_coo.device
    # convert to tuple # indices: int64 -> int16
    x_coo_tuple = (x_coo_indices, x_coo_values, x_coo_size, x_coo_device)
    return x_coo_tuple


def convert_tuple_to_coo(x_coo_tuple):
    return x_coo_tuple
    x_coo_indices = x_coo_tuple[0].to(torch.int64)
    x_coo_values = x_coo_tuple[1]
    x_coo_size = x_coo_tuple[2]
    x_coo_device = x_coo_tuple[3]
    x_coo = torch.sparse_coo_tensor(
        indices=x_coo_indices,
        values=x_coo_values,
        size=x_coo_size,
        device=x_coo_device
    )
    return x_coo


def fake_divide_outlier_suboutlier_svd(x: torch.Tensor, outlier: float, max_norm_column_list: float, scale: float, rank: int, sub_outlier_bit: int = 8, sub_outlier_ratio: float = 1.):
    is_head = len(x.shape) == 4
    if is_head:
        num_heads = x.shape[1]
        x = head_to_hidden_shape(x)
    
    # step 1: prune the outlier
    mask_1 = (x.abs() > outlier)
    x_outlier = x * mask_1
    x = x - x_outlier
    
    # step 2: prune the suboutlier
    if sub_outlier_ratio == 0.:
        x_sub_outlier = 0.
    else:
        if sub_outlier_ratio < 1.:
            mask_2 = torch.zeros_like(x).to(x.device).to(x.dtype)
            mask_2[:, :, max_norm_column_list] = 1
            mask_2 = mask_2.bool()

            x_sub_outlier = x * mask_2
            x = x - x_sub_outlier
        else:
            x_sub_outlier = x
        x_sub_outlier = torch.clamp(torch.round(x_sub_outlier / (scale + 1e-10)), min=-(2 ** (sub_outlier_bit - 1)), max=2 ** (sub_outlier_bit - 1) - 1) * scale
    
    # step 3: apply SVD
    if rank > 0:
        x = fake_svd_lowrank_simple_tensor(x, rank)
        x = x + x_outlier + x_sub_outlier
    else:
        x = x_outlier + x_sub_outlier
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    return x


@torch.no_grad
def true_divide_outlier_suboutlier_svd_compress(x: torch.Tensor, outlier: float, scale: float, sub_outlier_bit: int = 8, sub_outlier_ratio: float = 1., L: torch.Tensor = None, R: torch.Tensor = None):
    is_head = len(x.shape) == 4
    if is_head:
        num_heads = x.shape[1]
        x = head_to_hidden_shape(x)
    
    # step 1: substract the svd base
    tgt_L = torch.zeros((x.shape[-2], L.shape[-1]))
    x = x - (pad_cut_L(L, tgt_L) @ R)
    # x = low_rank_addition(pad_cut_L(L, tgt_L), -R, x)
    
    # step 2: prune the outlier
    mask_1 = (x.abs() > outlier)
    x_outlier = x * mask_1
    x = x - x_outlier
    # compress the x_outlier
    if torch.sum(scale) != 1.:
        x_outlier_compressed = convert_coo_to_tuple(x_outlier.to(torch.bfloat16).to_sparse()) # coo
    else:
        x_outlier_compressed = x_outlier
    del x_outlier
    
    # step 3: quantize the suboutlier
    if sub_outlier_ratio == 0.:
        x_sub_outlier = torch.tensor(0.).cuda()
        x_sub_outlier_compressed = torch.tensor(0.).cuda()
        scale = torch.tensor(1.).cuda()
    else:
        x_sub_outlier = x
        assert (sub_outlier_bit in [1, 2, 4, 8, 16]), "Only support 1,2,4,8,16 bit quantization"
        if sub_outlier_bit == 16:
            pass
        else:
            x_sub_outlier = torch.clamp(torch.round(x_sub_outlier / scale), min=-(2 ** (sub_outlier_bit - 1)), max=2 ** (sub_outlier_bit - 1) - 1)
            # now the x_sub_outlier is int type, then we can use bit squeeze method
            # since the shape is [bs, seq_len, hidden_dim], and hidden_dim is usually divisble by 8, so use hidden_dim dim to squeeze
            hidden_dim = x_sub_outlier.shape[-1]
            
            if sub_outlier_bit == 8:
                x_sub_outlier_compressed = x_sub_outlier.to(torch.int8)
            elif sub_outlier_bit == 4:
                # shift to positive
                x_sub_outlier = (x_sub_outlier + 8).to(torch.uint8)
                x_sub_outlier_compressed = x_sub_outlier[..., 0:(hidden_dim // 2)] \
                + x_sub_outlier[..., (hidden_dim // 2):] * (2 ** 4)
            elif sub_outlier_bit == 2:
                x_sub_outlier = (x_sub_outlier + 2).to(torch.uint8)
                x_sub_outlier_compressed = x_sub_outlier[..., ((hidden_dim // 4) * 3):hidden_dim] * (2 ** 6)
                x_sub_outlier_compressed += x_sub_outlier[..., (hidden_dim // 2):((hidden_dim // 4) * 3)] * (2 ** 4)
                x_sub_outlier_compressed += x_sub_outlier[..., (hidden_dim // 4):(hidden_dim // 2)] * (2 ** 2)
                x_sub_outlier_compressed += x_sub_outlier[..., 0:(hidden_dim // 4)]
            elif sub_outlier_bit == 1:
                x_sub_outlier = (x_sub_outlier + 1).to(torch.uint8)
                x_sub_outlier_compressed = x_sub_outlier[..., ((hidden_dim // 8) * 7):hidden_dim] * (2 ** 7)
                x_sub_outlier_compressed += x_sub_outlier[..., ((hidden_dim // 8) * 6):((hidden_dim // 8) * 7)] * (2 ** 6)
                x_sub_outlier_compressed += x_sub_outlier[..., ((hidden_dim // 8) * 5):((hidden_dim // 8) * 6)] * (2 ** 5)
                x_sub_outlier_compressed += x_sub_outlier[..., ((hidden_dim // 8) * 4):((hidden_dim // 8) * 5)] * (2 ** 4)
                x_sub_outlier_compressed += x_sub_outlier[..., ((hidden_dim // 8) * 3):((hidden_dim // 8) * 4)] * (2 ** 3)
                x_sub_outlier_compressed += x_sub_outlier[..., ((hidden_dim // 8) * 2):((hidden_dim // 8) * 3)] * (2 ** 2)
                x_sub_outlier_compressed += x_sub_outlier[..., ((hidden_dim // 8) * 1):((hidden_dim // 8) * 2)] * (2 ** 1)
                x_sub_outlier_compressed += x_sub_outlier[..., 0:(hidden_dim // 8)]
            del x_sub_outlier
    
    return x_outlier_compressed, x_sub_outlier_compressed, scale


@torch.no_grad
def true_divide_outlier_suboutlier_svd_decompress(x_outlier_compressed, x_sub_outlier_compressed, sub_outlier_bit, scale, is_head = False, num_heads = 1, L = None, R = None):
    x_outlier = convert_tuple_to_coo(x_outlier_compressed).to_dense()
    
    # step 1: add the base
    tgt_L = torch.zeros((x_outlier.shape[-2], L.shape[-1]))
    x = (pad_cut_L(L, tgt_L) @ R).to(torch.bfloat16)
    del tgt_L
    
    # step 2: add the outliers
    x = x + x_outlier
   
    # step 3: decompress the sub_outliers
    if torch.sum(scale) != 1.:
        if sub_outlier_bit == 16:
            x_sub_outlier = x_sub_outlier_compressed
        elif sub_outlier_bit == 8:
            # just return to the original value
            x_sub_outlier = x_sub_outlier_compressed.to(x_outlier.dtype) * scale
        elif sub_outlier_bit == 4:
            x_sub_outlier_1st = x_sub_outlier_compressed % (2 ** 4)
            x_sub_outlier_2nd = (x_sub_outlier_compressed - x_sub_outlier_1st) // (2 ** 4)
            x_sub_outlier = torch.cat((x_sub_outlier_1st, x_sub_outlier_2nd), dim=-1)
            del x_sub_outlier_1st, x_sub_outlier_2nd
            x_sub_outlier = ((x_sub_outlier).to(x_outlier.dtype) - 8) * scale
        elif sub_outlier_bit == 2:
            x_sub_outlier_1st = x_sub_outlier_compressed % (2 ** 2)
            x_sub_outlier_compressed = (x_sub_outlier_compressed - x_sub_outlier_1st) // (2 ** 2)
            x_sub_outlier_2nd = x_sub_outlier_compressed % (2 ** 2)
            x_sub_outlier_compressed = (x_sub_outlier_compressed - x_sub_outlier_2nd) // (2 ** 2)
            x_sub_outlier_3rd = x_sub_outlier_compressed % (2 ** 2)
            x_sub_outlier_4th = (x_sub_outlier_compressed - x_sub_outlier_3rd) // (2 ** 2)
            x_sub_outlier = torch.cat((x_sub_outlier_1st, x_sub_outlier_2nd, x_sub_outlier_3rd, x_sub_outlier_4th), dim=-1)
            del x_sub_outlier_1st, x_sub_outlier_2nd, x_sub_outlier_3rd, x_sub_outlier_4th
            x_sub_outlier = ((x_sub_outlier).to(x_outlier.dtype) - 2) * scale
        elif sub_outlier_bit == 1:
            x_sub_outlier_1st = x_sub_outlier_compressed % (2 ** 1)
            x_sub_outlier_compressed = (x_sub_outlier_compressed - x_sub_outlier_1st) // (2 ** 1)
            x_sub_outlier_2nd = x_sub_outlier_compressed % (2 ** 1)
            x_sub_outlier_compressed = (x_sub_outlier_compressed - x_sub_outlier_2nd) // (2 ** 1)
            x_sub_outlier_3rd = x_sub_outlier_compressed % (2 ** 1)
            x_sub_outlier_compressed = (x_sub_outlier_compressed - x_sub_outlier_3rd) // (2 ** 1)
            x_sub_outlier_4th = x_sub_outlier_compressed % (2 ** 1)
            x_sub_outlier_compressed = (x_sub_outlier_compressed - x_sub_outlier_4th) // (2 ** 1)
            x_sub_outlier_5th = x_sub_outlier_compressed % (2 ** 1)
            x_sub_outlier_compressed = (x_sub_outlier_compressed - x_sub_outlier_5th) // (2 ** 1)
            x_sub_outlier_6th = x_sub_outlier_compressed % (2 ** 1)
            x_sub_outlier_compressed = (x_sub_outlier_compressed - x_sub_outlier_6th) // (2 ** 1)
            x_sub_outlier_7th = x_sub_outlier_compressed % (2 ** 1)
            x_sub_outlier_8th = (x_sub_outlier_compressed - x_sub_outlier_7th) // (2 ** 1)
            x_sub_outlier = torch.cat((x_sub_outlier_1st, x_sub_outlier_2nd, x_sub_outlier_3rd, x_sub_outlier_4th, x_sub_outlier_5th, x_sub_outlier_6th, x_sub_outlier_7th, x_sub_outlier_8th), dim=-1)
            del x_sub_outlier_1st, x_sub_outlier_2nd, x_sub_outlier_3rd, x_sub_outlier_4th, x_sub_outlier_5th, x_sub_outlier_6th, x_sub_outlier_7th, x_sub_outlier_8th
            x_sub_outlier = ((x_sub_outlier).to(x_outlier.dtype) - 1) * scale
        x = x + x_sub_outlier

    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x


@torch.no_grad
def true_compress_softmax(x: torch.Tensor, outlier: float):
    mask = (x > outlier)
    x_outlier = x * mask
    x_outlier_sparse = x_outlier.to_sparse()
    x_outlier_sparse = convert_coo_to_tuple(x_outlier_sparse)
    return x_outlier_sparse


@torch.no_grad
def true_decompress_softmax(x_sparse: torch.Tensor):
    return convert_tuple_to_coo(x_sparse).to_dense()


def prune_softmax(x: torch.Tensor, outlier: float):
    mask = (x > outlier)
    x_outlier = x * mask
    return x_outlier


def profile_memory(name):
    print(f'---------------------{name}---------------------')
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
    print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    print('--------------------------------------------------')

@torch.no_grad
def get_statistics(x: torch.Tensor, iteration: int, outlier_ratio: float, sub_outlier_ratio: float, sub_outlier_bit: int = 8, sub_outlier_quantize_method: str = 'per-tensor', svd_rank: int = 16):    
    if len(x.shape) == 4:
        batch, num_head, seq_len, sep_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)
    if svd_rank > 0:
        # profile_memory('before_svd')
        with torch.no_grad():
            U, S, V = torch.svd_lowrank(x[0].to(torch.float32), q=svd_rank, niter=16)
            L = U
            L = L.contiguous()
            R = torch.diag(S) @ V.T
            R = R.contiguous()
            # profile_memory('before L @ R')
            x = x - L @ R
            # x = low_rank_addition(L, -R, x)
            # profile_memory('after L @ R')
            del U, S, V
        # print(L.shape, R.shape)
        # profile_memory('after_svd')
    else:
        # generate empty L and R such that the shape is consistent with the input
        # profile_memory('before_svd')
        L = torch.zeros((x.shape[-2], 16)).to(x.device).to(x.dtype)
        R = torch.zeros((16, x.shape[-1])).to(x.device).to(x.dtype)
        # profile_memory('after_svd')

    outlier = torch.kthvalue(x[0].flatten().to(torch.float32), int(x[0].numel() * (1 - outlier_ratio))).values
    x_outlier = x[0] * (x[0].abs() > outlier)
    x_outlier = x_outlier.to(torch.bfloat16)
    x = x - x_outlier
    
    if sub_outlier_ratio > 0 and sub_outlier_bit != 16:
        x_sub_outlier = x[0]
        if sub_outlier_quantize_method == 'per-tensor':
            # TODO: set the scale factor to per channel or per tensor?
            scale = (x_sub_outlier.max() - x_sub_outlier.min()) / (2 ** sub_outlier_bit - 1)
        elif sub_outlier_quantize_method == 'per-channel':
            # channel dimension: -2
            scale = (x_sub_outlier.max(dim=-2, keepdim=True).values - x_sub_outlier.min(dim=-2, keepdim=True).values) / (2 ** sub_outlier_bit - 1)
        elif sub_outlier_quantize_method == 'per-token':
            # token dimension: -1
            scale = (x_sub_outlier.max(dim=-1, keepdim=True).values - x_sub_outlier.min(dim=-1, keepdim=True).values) / (2 ** sub_outlier_bit - 1)
        else:
            raise "Unsupport Quantize Method"
    else:
        scale = torch.tensor(1.).cuda()
    
    return outlier, L.to(torch.bfloat16), R.to(torch.bfloat16), scale.to(torch.bfloat16)


@torch.no_grad
def get_statistics_softmax(x: torch.Tensor, iteration: int, outlier_ratio: float):
    outlier = torch.kthvalue(x[0].float().flatten(), int(x[0].numel() * (1 - outlier_ratio))).values
    # print(f'iter {iteration} | outlier: {outlier}')
    return outlier


@torch.no_grad
def pad_cut_L(src_L, tgt_L):
    # src_L: [seq_len_1, r]
    # tgt_L: [seq_len_2, r]
    seq_len_1, r = src_L.shape
    seq_len_2, _ = tgt_L.shape
    if seq_len_1 < seq_len_2:
        src_L = torch.cat((src_L, torch.zeros(seq_len_2 - seq_len_1, r).to(src_L.dtype).to(src_L.device)), dim=0)
    elif seq_len_1 > seq_len_2:
        src_L = src_L[0:seq_len_2, :]
    return src_L.contiguous()
