#!/bin/bash

# custom config
DATA=/data0/zhur/data
TRAINER=CoOp

DATASET=$1
CFG=vit_b16_ep50
SHOTS=$2
SUB=$3
SEED=$4 # random seed
DEVICE=$5
LOADEP=50

DIR=output_openworld/CoOp/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}
MODEL_DIR=output_openworld/CoOp/train/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}
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
    DATASET.NUM_SHOTS ${SHOTS}\
    DATASET.SUBSAMPLE_CLASSES ${SUB}

