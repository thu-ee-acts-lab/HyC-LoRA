import os
import torch
import matplotlib.pyplot as plt
from operators.compress_function import get_statistics, true_divide_outlier_suboutlier_svd_compress, true_divide_outlier_suboutlier_svd_decompress


def get_compress_ratio(b, s, r, w, p, d):
    compress_ratio = (b * s * d * 16) / ((s + d) * r * 16 + b * min(4 * p * s * d * 16, p * s * d * 16 + s * d) + s * d * b * w)
    return compress_ratio


if __name__ == '__main__':
    data_kinds = [
        'input_layernorm.pt', 
        # 'self_attn.q_proj.lora_A.default.pt', 
        'self_attn.gemm1_1.pt',
        'self_attn.gemm1_2.pt',
        # 'self_attn.gemm2_2.pt',
        # 'self_attn.o_proj.lora_A.default.pt',
        'post_attention_layernorm.pt',
        # 'mlp.up_proj.lora_A.default.pt',
        'mlp.act_fn.pt',
        'mlp.hadamard_1.pt',
        'mlp.hadamard_2.pt',
        # 'mlp.down_proj.lora_A.default.pt',
    ]
    
    for layer in range(32):
        if not os.path.exists(f'picture_new/{layer}'):
            os.mkdir(f'picture_new/{layer}')
        
        for data_kind in data_kinds:
            plt.figure(figsize=(20, 20))
            data_raw = torch.load(f'/home/yujin-wa20/projects/LoftQ/final_feature/base_model.model.model.layers.{layer}.{data_kind}')
            data_raw = data_raw.cuda().to(torch.bfloat16)
            
            mean = data_raw.mean().item()
            std = data_raw.std().item()
            
            # construct the data
            data_gaussian = torch.randn_like(data_raw) * std + mean
            
            for svd_rank in [0, 16]:
                for sub_outlier_bit in [1, 2, 4]:
                    compress_ratio_list = []
                    mse_list = []
                    mse_gaussian_list = []
                    for outlier_ratio in [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
                        print('-------------------------------------------------------------------')
                        print(f'sub_outlier_bit: {sub_outlier_bit}, outlier_ratio: {outlier_ratio}')
                
                        # quantize data
                        outlier, L, R, scale = get_statistics(data_raw[0:4], 1, outlier_ratio, 1, sub_outlier_bit, 'per-channel', svd_rank)
                        data = data_raw[4:8]
                        x_outlier_compressed, x_sub_outlier_compressed, scale = true_divide_outlier_suboutlier_svd_compress(data, outlier, scale, sub_outlier_bit, 1, L, R)
                        
                        outlier_gaussian, L_gaussian, R_gaussian, scale_gaussian = get_statistics(data_gaussian, 1, outlier_ratio, 1, sub_outlier_bit, 'per-channel', svd_rank)
                        x_outlier_compressed_gaussian, x_sub_outlier_compressed_gaussian, scale_gaussian = true_divide_outlier_suboutlier_svd_compress(data_gaussian, outlier_gaussian, scale_gaussian, sub_outlier_bit, 1, L_gaussian, R_gaussian)
                        
                        is_head = len(data.size()) == 4
                        if is_head:
                            head_num = data.size(1)
                        else:
                            head_num = 1
                        
                        x_decompressed = true_divide_outlier_suboutlier_svd_decompress(x_outlier_compressed, x_sub_outlier_compressed, sub_outlier_bit, scale, is_head, head_num, L, R)
                        x_decompressed_gaussian = true_divide_outlier_suboutlier_svd_decompress(x_outlier_compressed_gaussian, x_sub_outlier_compressed_gaussian, sub_outlier_bit, scale_gaussian, is_head, head_num, L_gaussian, R_gaussian)
                        
                        # calculate the mse
                        mse = torch.nn.functional.mse_loss(data, x_decompressed)
                        mse_gaussian = torch.nn.functional.mse_loss(data_gaussian, x_decompressed_gaussian)
                        # print the config and mse
                        print(mse.item())
                        print(mse_gaussian.item())
                        
                        # get compress ratio
                        compress_ratio = get_compress_ratio(4, data.size(1), svd_rank, sub_outlier_bit, outlier_ratio, data.size(2))
                        print(compress_ratio)
                        
                        compress_ratio_list.append(compress_ratio)
                        mse_list.append(mse.item())
                        mse_gaussian_list.append(mse_gaussian.item())
                
                    plt.plot(compress_ratio_list, mse_list, label=f'sub_outlier_bit: {sub_outlier_bit}, rank: {svd_rank}, {mse_list[0] / mse_list[5]}', marker='o')
                    # another marker type
                    plt.plot(compress_ratio_list, mse_gaussian_list, label=f'sub_outlier_bit: {sub_outlier_bit} (gaussian), rank: {svd_rank}, {mse_gaussian_list[0] / mse_gaussian_list[5]}', marker='x')
            
            # set y to log scale
            plt.yscale('log')
            plt.xlabel('compress ratio')
            plt.ylabel('mse')
            plt.legend()
            plt.savefig(f'picture_new/{layer}/{data_kind}.png')
        