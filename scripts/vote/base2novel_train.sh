DATA=/data0/zhur/data
TRAINER=Vote

DATASET=$1
SHOTS=$2
SEED=$3
DEVICE=$4

DIR=output/${TRAINER}/base2novel_train/${DATASET}/shots_${SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=$DEVICE python vote_main.py \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.MODAL  base2novel \
    TRAIN_OR_TEST train