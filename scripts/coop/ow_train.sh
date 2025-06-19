# 在 base 类的训练集上进行训练，选择在 base 类的验证集上精度最高的模型作为最终的模型，并在 base 类的测试集上进行测试

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


DIR=output_openworld/CoOp/train/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}
CUDA_VISIBLE_DEVICES=$DEVICE python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}\
    DATASET.SUBSAMPLE_CLASSES ${SUB}

