import torch
import argparse
from operators.compress_function import get_statistics, true_divide_outlier_suboutlier_svd_compress, true_divide_outlier_suboutlier_svd_decompress

def get_compress_ratio(b, s, r, w, p, d):
    compress_ratio = (b * s * d * 16) / ((s + d) * r * 16 + b * min(4 * p * s * d * 16, p * s * d * 16 + s * d) + s * d * b * w)
    print('compress ratio: {:.2f}'.format(compress_ratio))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outlier-ratio', type=float, default=0.01)
    parser.add_argument('--sub-outlier-bit', type=int, default=2)
    parser.add_argument('--svd-rank', type=int, default=0)
    
    args = parser.parse_args()
    
    data = torch.load('/home/yujin-wa20/projects/LoftQ/output/mistral/base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.pt')
    data = data.cuda().to(torch.bfloat16)
    
    # quantize data
    outlier, L, R, scale = get_statistics(data, 1, args.outlier_ratio, 1, args.sub_outlier_bit, 'per-channel', args.svd_rank)
    x_outlier_compressed, x_sub_outlier_compressed, scale = true_divide_outlier_suboutlier_svd_compress(data, outlier, scale, args.sub_outlier_bit, 1, L, R)
    x_decompressed = true_divide_outlier_suboutlier_svd_decompress(x_outlier_compressed, x_sub_outlier_compressed, args.sub_outlier_bit, scale, False, 1, L, R)
    
    # calculate the mse
    mse = torch.nn.functional.mse_loss(data, x_decompressed)
    # print the config and mse
    print(args)
    print(mse.item())
    
    # get compress ratio
    get_compress_ratio(4, data.size(1), args.svd_rank, args.sub_outlier_bit, args.outlier_ratio, data.size(2))
    