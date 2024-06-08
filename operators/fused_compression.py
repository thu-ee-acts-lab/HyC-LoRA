import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8, }, num_stages=4, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def low_rank_extraction_kernel(
    # Pointers to matrices
    l_ptr, r_ptr, x_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_lm, stride_lk,
    stride_rk, stride_rn,
    stride_xb, stride_xm, stride_xn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_lm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    l_ptrs = l_ptr + (offs_lm[:, None] * stride_lm + offs_k[None, :] * stride_lk)
    r_ptrs = r_ptr + (offs_k[:, None] * stride_rk + offs_rn[None, :] * stride_rn)

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
    accumulator = tl.load(x_ptrs, mask=x_mask, other=0.0)
    accumulator = accumulator.to(tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(l_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(r_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator -= tl.dot(a, b)
        # Advance the ptrs to the next K block.
        l_ptrs += BLOCK_SIZE_K * stride_lk
        r_ptrs += BLOCK_SIZE_K * stride_rk

    # divide the simple values and outlier values
    x = accumulator.to(tl.bfloat16)
    tl.store(x_ptrs, x, mask=x_mask)


'''
input:
l: [s, r]
r: [r, d]
x: [b, s, d]

output:
q: [b, s, d // n], quantized version
(Notice x convert to sparse format inplacely)
'''
def low_rank_extraction(l, r, x):
    # Check constraints.
    assert l.shape[1] == r.shape[0], "Incompatible dimensions"
    assert l.is_contiguous(), "Matrix A must be contiguous"
    assert r.is_contiguous(), "Matrix B must be contiguous"
    M, K = l.shape
    K, N = r.shape
    B, _, _ = x.shape
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    low_rank_extraction_kernel[grid](
        l, r, x,
        B, M, N, K,
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_SIZE_K=K
    )
    
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8, }, num_stages=4, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def low_rank_addition_kernel(
    # Pointers to matrices
    l_ptr, r_ptr, x_ptr, o_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_lm, stride_lk,
    stride_rk, stride_rn,
    stride_xb, stride_xm, stride_xn,
    stride_ob, stride_om, stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_lm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    l_ptrs = l_ptr + (offs_lm[:, None] * stride_lm + offs_k[None, :] * stride_lk)
    r_ptrs = r_ptr + (offs_k[:, None] * stride_rk + offs_rn[None, :] * stride_rn)

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
    
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    o_ptrs = o_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :] + offs_b * stride_ob
    o_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
    
    accumulator = tl.load(x_ptrs, mask=x_mask, other=0.0)
    accumulator = accumulator.to(tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(l_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(r_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        l_ptrs += BLOCK_SIZE_K * stride_lk
        r_ptrs += BLOCK_SIZE_K * stride_rk

    # divide the simple values and outlier values
    x = accumulator.to(tl.bfloat16)
    tl.store(o_ptrs, x, mask=o_mask)


'''
input:
l: [s, r]
r: [r, d]
x: [b, s, d]

output:
q: [b, s, d // n], quantized version
(Notice x convert to sparse format inplacely)
'''
def low_rank_addition(l, r, x):
    # Check constraints.
    assert l.shape[1] == r.shape[0], "Incompatible dimensions"
    assert l.is_contiguous(), "Matrix A must be contiguous"
    assert r.is_contiguous(), "Matrix B must be contiguous"
    M, K = l.shape
    K, N = r.shape
    B, _, _ = x.shape
    if K < 16:
        l = torch.cat([l, torch.zeros((M, 16 - K), device=l.device, dtype=l.dtype)], dim=1).contiguous()
        r = torch.cat([r, torch.zeros((16 - K, N), device=r.device, dtype=r.dtype)], dim=0).contiguous()
        K = 16
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    o = torch.empty((B, M, N), device=x.device, dtype=torch.bfloat16)
    low_rank_addition_kernel[grid](
        l, r, x, o,
        B, M, N, K,
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        x.stride(0), x.stride(1), x.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_SIZE_K=K
    )
    return o


if __name__ == '__main__':
    x = torch.randn((2, 1024, 1024)).cuda().to(torch.bfloat16)
    l = torch.randn((1024, 8)).cuda().to(torch.bfloat16)
    r = torch.randn((8, 1024)).cuda().to(torch.bfloat16)
    print(torch.cuda.memory_allocated() / 1024 / 1024)
    x = low_rank_addition(l, r, x)
    print(torch.cuda.memory_allocated() / 1024 / 1024)