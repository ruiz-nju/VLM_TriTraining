DATA=/mnt/hdd/zhurui/data
TRAINER=TriTraining

DATASET=$1
SHOTS=$2
UNLABELED_SHOTS=$3
SEED=$4

# sh scripts/tritraining/base2novel_train.sh caltech101 16 16 1
DIR=output/${TRAINER}/base2novel_train/${DATASET}/shots_${SHOTS}/unlabeled_shots_${UNLABELED_SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=1 python tritraining_main.py \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.NUM_UNLABELED_SHOTS ${UNLABELED_SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.MODAL  base2novel