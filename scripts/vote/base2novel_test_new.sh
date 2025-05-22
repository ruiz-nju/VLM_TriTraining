DATA=/data0/zhur/data
TRAINER=Vote

DATASET=$1
SHOTS=$2
SEED=$3
DEVICE=$4

MODEL_DIR=output/${TRAINER}/base2novel_train/${DATASET}/shots_${SHOTS}/seed_${SEED}
DIR=output/${TRAINER}/base2novel_test_new/${DATASET}/shots_${SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=$DEVICE python vote_main.py \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES new \
    TRAINER.MODAL  base2novel \
    TRAIN_OR_TEST test