SAVE_DIR="model_zoo/loftq"
MODEL_PATH="~/projects/llama3/Meta-Llama-3-8B-HF"

python generate_loftq.py \
    --model_name_or_path  \
    --bits 4 \
    --iter 5 \
    --rank 16 \
    --save_dir $SAVE_DIR