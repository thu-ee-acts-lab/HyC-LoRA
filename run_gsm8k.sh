# finetune
set -x
svd_rank=2
outlier_ratio=0.002
softmax_outlier_ratio=0.02
sub_outlier_ratio=1
sub_outlier_bit=8
sub_outlier_quant_method=per-channel
lr=3e-4
gradient_accumulation_steps=4

echo $tag

for outlier_ratio in 0.001
do
    tag=Mistral-7B-v0.1-4bit-16rank-svd-rank-${svd_rank}-outlier_ratio-${outlier_ratio}-sub_outlier_ratio-${sub_outlier_ratio}-softmax_outlier_ratio-${softmax_outlier_ratio}-sub_outlier_bit-${sub_outlier_bit}-sub_outlier_quant_method-${sub_outlier_quant_method}-lr-${lr}-gradient_accumulation_steps-${gradient_accumulation_steps}
    exp_name=gsm8k_${tag}
    model_name_kind=model_zoo/loftq
    model_name_small=Mistral-7B-v0.1-4bit-16rank
    model_name=${model_name_kind}/${model_name_small}

    python -u main.py \
        --model_name_or_path /home/yujin-wa20/projects/LoftQ/${model_name} \
        --data_name gsm8k \
        --bits 4 \
        --learning_rate $lr \
        --seed 11 \
        --expt_name $exp_name \
        --output_dir exp_results/$exp_name/ \
        --num_train_epochs 6 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --weight_decay 0.1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --do_train \
        --transform_bp_enable \
        --linear_outlier_ratio $outlier_ratio \
        --linear_sub_outlier_ratio $sub_outlier_ratio \
        --linear_sub_outlier_bit $sub_outlier_bit \
        --linear_rank $svd_rank \
        --linear_sub_outlier_quant_method $sub_outlier_quant_method \
        --silu_outlier_ratio $outlier_ratio \
        --silu_sub_outlier_ratio $sub_outlier_ratio \
        --silu_sub_outlier_bit $sub_outlier_bit \
        --silu_rank $svd_rank \
        --silu_sub_outlier_quant_method $sub_outlier_quant_method \
        --layernorm_outlier_ratio $outlier_ratio \
        --layernorm_sub_outlier_ratio $sub_outlier_ratio \
        --layernorm_sub_outlier_bit $sub_outlier_bit \
        --layernorm_rank $svd_rank \
        --layernorm_sub_outlier_quant_method $sub_outlier_quant_method \
        --hadamard_outlier_ratio $outlier_ratio \
        --hadamard_sub_outlier_ratio $sub_outlier_ratio \
        --hadamard_sub_outlier_bit $sub_outlier_bit \
        --hadamard_rank $svd_rank \
        --hadamard_sub_outlier_quant_method $sub_outlier_quant_method \
        --gemm_outlier_ratio $outlier_ratio \
        --gemm_sub_outlier_ratio $sub_outlier_ratio \
        --gemm_sub_outlier_bit $sub_outlier_bit \
        --gemm_rank $svd_rank \
        --gemm_sub_outlier_quant_method $sub_outlier_quant_method \
        --softmax_outlier_ratio $softmax_outlier_ratio \
        --softmax_sub_outlier_ratio $sub_outlier_ratio \
        --softmax_sub_outlier_bit $sub_outlier_bit \
        --softmax_rank $svd_rank

    # python test_gsm8k.py \
       # --model_name_or_path /home/yujin-wa20/projects/LoftQ/model_zoo/${model_name} \
       # --adapter_name_or_path /home/yujin-wa20/projects/LoftQ/exp_results/${exp_name}/${exp_name}/${model_name_small}/ep_6/lr_0.0003/seed_11 \
       # --batch_size 16

    echo $tag
done
     
