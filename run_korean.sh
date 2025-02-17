export MODEL=wav2vec2-large-xlsr-korean
export TOKENIZER=wav2vec2-large-xlsr-korean
export ALPHA=0.1
export LR=5e-5
export ACC=4 # batch size * acc = 8
export WORKER_NUM=4

/home/jovyan/YJ_DATA/miniconda3/envs/py37/bin/python run_emotion.py \
--output_dir=output/tmp \
--cache_dir=cache_kor/ \
--num_train_epochs=200 \
--per_device_train_batch_size="2" \
--per_device_eval_batch_size="2" \
--gradient_accumulation_steps=$ACC \
--alpha $ALPHA \
--dataset_name kemdy19 \
--split_id 01F \
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
--orthography korean
# --freeze_feature_extractor \
# --gradient_checkpointing true \
# --fp16 \
