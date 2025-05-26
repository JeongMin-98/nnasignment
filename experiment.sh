#!/bin/bash

# 실험 설정
EPOCHS=40
BATCH_SIZE=32
LR=0.001
USE_WANDB="--use_wandb"  # WANDB 사용 안할 경우 빈 문자열로 설정

# 실험 대상 모델과 옵티마이저 조합
MODELS=("resnet18" "resnet34" "resnet50")
OPTIMIZERS=("sgd" "adam")

# 순차적 실행
for model in "${MODELS[@]}"; do
  for opt in "${OPTIMIZERS[@]}"; do
    echo "🚀 실행 중: 모델=${model}, 옵티마이저=${opt}"
    python main.py \
      --model $model \
      --optimizer $opt \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --lr $LR \
      $USE_WANDB

    echo "✅ 완료: ${model}-${opt}"
    echo "--------------------------"
  done
done

echo "🎉 모든 실험 완료!"
