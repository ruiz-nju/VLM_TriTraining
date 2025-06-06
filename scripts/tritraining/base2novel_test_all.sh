DATA=/data0/zhur/data
TRAINER=TriTraining

DATASET=$1
SHOTS=$2
UNLABELED_SHOTS=$3
SEED=$4
DEVICE=$5

MODEL_DIR=output/${TRAINER}/base2novel_train/${DATASET}/shots_${SHOTS}/unlabeled_shots_${UNLABELED_SHOTS}/seed_${SEED}
DIR=output/${TRAINER}/base2novel_test_all/${DATASET}/shots_${SHOTS}/unlabeled_shots_${UNLABELED_SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=$DEVICE python tritraining_main.py \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.NUM_UNLABELED_SHOTS ${UNLABELED_SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all \
    TRAINER.MODAL  base2novel \
    TRAIN_OR_TEST test