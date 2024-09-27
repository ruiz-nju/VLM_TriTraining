#!/bin/bash

# custom config
DATA=/mnt/hdd/zhurui/data
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
SEED=$4 # random seed

CTP=$5  # class token position (end or middle)
NCTX=$6  # number of context tokens
CSC=$7  # class-specific context (False or True)

DIR=output/${DATASET}/${TRAINER}/main/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed_${SEED}
CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}\
    # TRAINER.MODAL base2novel \

# TRAINER.COOP.N_CTX 默认为 16
# TRAINER.COOP.CSC 默认为 False
# TRAINER.COOP.CLASS_TOKEN_POSITION 默认为 end
# DATASET.NUM_SHOTS 是通过 opts 传入的，会在 setup_cfg 中通过 cfg.merge_from_list(args.opts) 来动态修改配置