#!/bin/bash

# 커맨드라인 인자 처리
SPLIT="${1:-fold1}"  # 기본값 fold1, 인자로 다른 값 지정 가능 (e.g., fold1, fold2, ...)
ALPHA="${2:-0.0}"  # 학습 시 사용했던 alpha 값과 동일하게 설정
DEVICE="${3:-auto}" # 기본값 auto (자동 감지), 인자로 다른 값 지정 가능

# 모델 및 데이터 설정 (run_korean.sh와 일치시키세요)
MODEL="wav2vec2-large-xlsr-korean"
TOKENIZER="wav2vec2-large-xlsr-korean"
DATASET="kemdy19"

# 학습된 모델 경로 설정 (run_korean.sh의 SAVE_PATH와 유사하게 구성)
# 주의: 실제 학습된 모델이 있는 경로로 정확히 지정해야 합니다.
MODEL_PATH="output/${DATASET}/${DATASET}_${MODEL}_alpha${ALPHA}_${SPLIT}"

# Interpretation 결과 저장 디렉토리 설정
INTERPRET_DIR="${MODEL_PATH}/interpretation_results"

# 디바이스 자동 감지 (auto인 경우)
if [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null
    then
        DEVICE="cuda"
    else
        DEVICE="cpu"
        echo "CUDA를 찾을 수 없습니다. CPU를 사용합니다."
    fi
fi

echo "==== Interpretation 계산 시작 (Attention Weights + IG) ===="
echo "모델 경로: ${MODEL_PATH}"
echo "데이터셋: ${DATASET}_${SPLIT}"
echo "결과 저장 디렉토리: ${INTERPRET_DIR}"
echo "장치: ${DEVICE}"

# run_emotion.py 실행 (interpretation 모드로)
# 주의: --model_name_or_path 는 학습된 모델 경로를 지정해야 합니다.
python run_emotion.py \
    --model_name_or_path="${MODEL_PATH}" \
    --cache_dir=cache_kor/ \
    --dataset_name="${DATASET}" \
    --split_id="fold${SPLIT}" \
    --output_dir="${MODEL_PATH}" \
    --per_device_eval_batch_size=1 \
    --preprocessing_num_workers=4 \
    --dataloader_num_workers=4 \
    --tokenizer="kresnik/${TOKENIZER}" \
    --orthography=korean \
    --alpha=${ALPHA} \
    --do_interpret \
    --interpretation_output_dir="${INTERPRET_DIR}" \
    --report_to="none" # 해석 중에는 wandb 로깅 비활성화

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "==== Interpretation 계산 완료 ===="
    echo "결과 저장 경로: ${INTERPRET_DIR}"

    # 결과 시각화 스크립트 실행 (선택적)
    echo "==== 결과 시각화 시작 ===="
    VIS_OUTPUT_DIR="${INTERPRET_DIR}/plots"
    python visualize_interpretations.py \
        --input_dir="${INTERPRET_DIR}" \
        --output_dir="${VIS_OUTPUT_DIR}" \
        --vis_type both
    echo "==== 결과 시각화 완료 ===="
    echo "시각화 결과 저장 경로: ${VIS_OUTPUT_DIR}"
else
    echo "==== Interpretation 계산 중 오류 발생 (종료 코드: $EXIT_CODE) ===="
fi 