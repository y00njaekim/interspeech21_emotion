export MODEL=wav2vec2-large-xlsr-korean
export TOKENIZER=wav2vec2-large-xlsr-korean
export ALPHA=0.0
export LR=5e-5
export ACC=1 # batch size * acc = 8
export WORKER_NUM=4
export SPLIT=$1 # split parameter (e.g., 01F, 01M)

# 현재 시간을 포함한 run 이름 생성 (한국어 실험용)
export RUN_NAME="kemdy19_${MODEL}_alpha${ALPHA}_${SPLIT}"
export SAVE_PATH="output/kemdy19/${RUN_NAME}"

/home/YJ_DATA/miniconda3/envs/py37/bin/python run_emotion.py \
--output_dir=$SAVE_PATH \
--cache_dir=cache_kor/ \
--overwrite_cache \
--overwrite_output_dir \
--num_train_epochs=100 \
--per_device_train_batch_size="8" \
--per_device_eval_batch_size="8" \
--gradient_accumulation_steps=$ACC \
--alpha $ALPHA \
--dataset_name kemdy19 \
--split_id fold${SPLIT} \
--evaluation_strategy="steps" \
--save_total_limit="1" \
--save_steps="500" \
--eval_steps="500" \
--logging_steps="50" \
--logging_dir="log" \
--do_train \
--do_eval \
--learning_rate=$LR \
--model_name_or_path=kresnik/$MODEL \
--tokenizer kresnik/$TOKENIZER \
--preprocessing_num_workers=$WORKER_NUM \
--dataloader_num_workers $WORKER_NUM \
--orthography korean \
--report_to="wandb" \
--wandb_project="emotion_recognition" \
--wandb_run_name="$RUN_NAME"
# --freeze_feature_extractor \
# --gradient_checkpointing true \
# --fp16 \
