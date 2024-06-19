import torch
import matplotlib.pyplot as plt
from operators.compress_function import get_statistics, true_divide_outlier_suboutlier_svd_compress, true_divide_outlier_suboutlier_svd_decompress


def get_compress_ratio(b, s, r, w, p, d):
    compress_ratio = (b * s * d * 16) / ((s + d) * r * 16 + b * min(4 * p * s * d * 16, p * s * d * 16 + s * d) + s * d * b * w)
    return compress_ratio


if __name__ == '__main__':
    # generate gaussian distribution of certain main and std
    mean = 0.0051
    std = 0.3945
    
    data = torch.randn((1, 320, 4096)) * std + mean
    data = data.cuda().to(torch.bfloat16)
    svd_rank = 0
    
    for sub_outlier_bit in [1, 2, 4]:
        compress_ratio_list = []
        mse_list = []
        for outlier_ratio in [0.00001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
            print('-------------------------------------------------------------------')
            print(f'sub_outlier_bit: {sub_outlier_bit}, outlier_ratio: {outlier_ratio}')
    
            # quantize data
            outlier, L, R, scale = get_statistics(data, 1, outlier_ratio, 1, sub_outlier_bit, 'per-channel', svd_rank)
            x_outlier_compressed, x_sub_outlier_compressed, scale = true_divide_outlier_suboutlier_svd_compress(data, outlier, scale, sub_outlier_bit, 1, L, R)
            x_decompressed = true_divide_outlier_suboutlier_svd_decompress(x_outlier_compressed, x_sub_outlier_compressed, sub_outlier_bit, scale, False, 1, L, R)
            
            # calculate the mse
            mse = torch.nn.functional.mse_loss(data, x_decompressed)
            # print the config and mse
            print(mse.item())
            
            # get compress ratio
            compress_ratio = get_compress_ratio(4, data.size(1), svd_rank, sub_outlier_bit, outlier_ratio, data.size(2))
            print(compress_ratio)
            
            compress_ratio_list.append(compress_ratio)
            mse_list.append(mse.item())
        
        plt.plot(compress_ratio_list, mse_list, label=f'sub_outlier_bit: {sub_outlier_bit}', marker='o')
    
    # set y to log scale
    plt.yscale('log')
    plt.xlabel('compress ratio')
    plt.ylabel('mse')
    plt.legend()
    plt.savefig('compress_ratio_mse_gaussian.png')