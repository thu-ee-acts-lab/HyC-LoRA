import torch

import triton
import triton.language as tl


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _seeded_dropout_backward(
    grad_out_ptr,
    grad_in_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from grad_out
    mask = offsets < n_elements
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    grad_out_keep = random > p
    # write-back
    grad_in = tl.where(grad_out_keep, grad_out, 0.0)
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


class EfficientMemoryDropoutFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, seed):
        ctx.p = p
        ctx.seed = seed
        output = torch.empty_like(x)
        assert x.is_contiguous()
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        p, seed = ctx.p, ctx.seed
        grad_in = torch.empty_like(grad_out)
        assert grad_out.is_contiguous()
        n_elements = grad_out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _seeded_dropout_backward[grid](
            grad_out, grad_in, n_elements, p, seed, BLOCK_SIZE=1024
        )
        return grad_in, None, None


class EfficientMemoryDropout(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        # randomly generate a seed(per layer)
        self.seed = torch.randint(0, 2**32, (1,)).item()

    def forward(self, x):
        # notice that dropout works differently in training and evaluation mode
        if self.training:
            return EfficientMemoryDropoutFunc.apply(x, self.p, self.seed)
        else:
            return x
