# 在 base 类的训练集上进行训练，选择在 base 类的验证集上精度最高的模型作为最终的模型，并在 base 类的测试集上进行测试

#!/bin/bash

# custom config
DATA=/data0/zhur/data
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file 例如 vit_b16
SHOTS=$3 
SEED=$4 # random seed
DEVICE=$5


DIR=output/reproduce/${DATASET}/${TRAINER}/base2novel_train/${CFG}/shots_${SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=$DEVICE python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}\
    TRAINER.MODAL base2novel \
    DATASET.SUBSAMPLE_CLASSES base \


# 通过 opts 传入的参数，会在 setup_cfg 中通过 cfg.merge_from_list(args.opts) 来动态修改配置