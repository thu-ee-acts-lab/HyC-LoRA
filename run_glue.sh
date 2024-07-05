task_name=mnli
svd_rank=2
outlier_ratio=0.01
softmax_outlier_ratio=0.05
sub_outlier_ratio=0
sub_outlier_bit=0
sub_outlier_quant_method=per-channel

for outlier_ratio in 0.1
    do
    for sub_outlier_bit in 2
    do
        for svd_rank in 0
        do
            for softmax_outlier_ratio in 0.05
            do
                python -u train_glue.py \
                    --model-name-or-path roberta-base \
                    --task-name $task_name \
                    --max-length 128 \
                    --per-device-train-batch-size 32 \
                    --per-device-eval-batch-size 128 \
                    --learning-rate 3e-4 \
                    --num-train-epochs 10 \
                    --seed 42 \
                    --output-dir log/$task_name \
                    --pad-to-max-length \
                    --linear-outlier-ratio $outlier_ratio \
                    --linear-sub-outlier-ratio $sub_outlier_ratio \
                    --linear-sub-outlier-bit $sub_outlier_bit \
                    --linear-rank $svd_rank \
                    --linear-sub-outlier-quant-method $sub_outlier_quant_method \
                    --silu-outlier-ratio $outlier_ratio \
                    --silu-sub-outlier-ratio $sub_outlier_ratio \
                    --silu-sub-outlier-bit $sub_outlier_bit \
                    --silu-rank $svd_rank \
                    --silu-sub-outlier-quant-method $sub_outlier_quant_method \
                    --layernorm-outlier-ratio $outlier_ratio \
                    --layernorm-sub-outlier-ratio $sub_outlier_ratio \
                    --layernorm-sub-outlier-bit $sub_outlier_bit \
                    --layernorm-rank $svd_rank \
                    --layernorm-sub-outlier-quant-method $sub_outlier_quant_method \
                    --gemm-outlier-ratio $outlier_ratio \
                    --gemm-sub-outlier-ratio $sub_outlier_ratio \
                    --gemm-sub-outlier-bit $sub_outlier_bit \
                    --gemm-rank $svd_rank \
                    --gemm-sub-outlier-quant-method $sub_outlier_quant_method \
                    --softmax-outlier-ratio $softmax_outlier_ratio \
                    --softmax-sub-outlier-ratio $sub_outlier_ratio \
                    --softmax-sub-outlier-bit $sub_outlier_bit \
                    --softmax-rank $svd_rank
            done
        done
    done
done

# sub_outlier_ratio=1

# for outlier_ratio in 0.00000001
#     do
#     for sub_outlier_bit in 1 2
#     do
#         for svd_rank in 0
#         do
#             for softmax_outlier_ratio in 0.05
#             do
#                 python -u train_glue.py \
#                     --model-name-or-path roberta-base \
#                     --task-name $task_name \
#                     --max-length 128 \
#                     --per-device-train-batch-size 32 \
#                     --per-device-eval-batch-size 128 \
#                     --learning-rate 3e-4 \
#                     --num-train-epochs 10 \
#                     --seed 42 \
#                     --output-dir log/$task_name \
#                     --pad-to-max-length \
#                     --linear-outlier-ratio $outlier_ratio \
#                     --linear-sub-outlier-ratio $sub_outlier_ratio \
#                     --linear-sub-outlier-bit $sub_outlier_bit \
#                     --linear-rank $svd_rank \
#                     --linear-sub-outlier-quant-method $sub_outlier_quant_method \
#                     --silu-outlier-ratio $outlier_ratio \
#                     --silu-sub-outlier-ratio $sub_outlier_ratio \
#                     --silu-sub-outlier-bit $sub_outlier_bit \
#                     --silu-rank $svd_rank \
#                     --silu-sub-outlier-quant-method $sub_outlier_quant_method \
#                     --layernorm-outlier-ratio $outlier_ratio \
#                     --layernorm-sub-outlier-ratio $sub_outlier_ratio \
#                     --layernorm-sub-outlier-bit $sub_outlier_bit \
#                     --layernorm-rank $svd_rank \
#                     --layernorm-sub-outlier-quant-method $sub_outlier_quant_method \
#                     --gemm-outlier-ratio $outlier_ratio \
#                     --gemm-sub-outlier-ratio $sub_outlier_ratio \
#                     --gemm-sub-outlier-bit $sub_outlier_bit \
#                     --gemm-rank $svd_rank \
#                     --gemm-sub-outlier-quant-method $sub_outlier_quant_method \
#                     --softmax-outlier-ratio $softmax_outlier_ratio \
#                     --softmax-sub-outlier-ratio $sub_outlier_ratio \
#                     --softmax-sub-outlier-bit $sub_outlier_bit \
#                     --softmax-rank $svd_rank
#             done
#         done
#     done
# done