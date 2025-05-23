#!/bin/bash


# custom config
DATA=/data0/zhur/data
TRAINER=PromptSRC

DATASET=$1
SEED=$2

CFG=vit_b16
SHOTS=$3
LOADEP=20
SUB=$4
DEVICE=$5

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output_openworld/base2new/train_base/${COMMON_DIR}
DIR=output_openworld/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    CUDA_VISIBLE_DEVICES=$DEVICE python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.NUM_LABELED ${SHOTS}\
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=$DEVICE python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.NUM_LABELED ${SHOTS}\
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi