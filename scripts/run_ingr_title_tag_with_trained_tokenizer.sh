# WANDB_MODE=offline \
    # --model_name_or_path roberta-base \
# CUDA_VISIBLE_DEVICES=1,2,3 \
    # --model_name_or_path bert-base-uncased \
# CUDA_VISIBLE_DEVICES=1,2,3 \
# CUDA_VISIBLE_DEVICES=1 \
#  --dataset_name wikitext \
# CUDA_VISIBLE_DEVICES=1,2,3 \
WANDB_PROJECT=ingt-v3-tagtokenizer \
WANDB_MODE=online \
CUDA_VISIBLE_DEVICES=0,1 \
python run_mlm.py \
    --run_name v3-ing-title-tag-tagtokenizer \
    --model_type bert \
    --tokenizer_name bert-base-uncased \
    --max_seq_length 25 \
    --train_file /disk1/data/ing_mlm_data/processed/v3_ing_title_tag/train.txt \
    --validation_file /disk1/data/ing_mlm_data/processed/v3_ing_title_tag/val.txt \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --do_train \
    --do_eval \
    --output_dir ./checkpoints/v3-ing-title-tagtokenizer \
    --overwrite_output_dir \
    --logging_steps 100 \
    --log_level info \
    --evaluation_strategy steps \
    --num_train_epochs 100 \
    --save_steps 5000 \
    --line_by_line true \
    --pad_to_max_length true \
    --config_overrides num_attention_heads=3,num_hidden_layers=3 \
    
    # --max_steps=200 \
    # --learning_rate



