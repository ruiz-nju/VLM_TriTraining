DATA=/data0/zhur/data
TRAINER=TriTraining

DATASET=$1
SHOTS=$2
UNLABELED_SHOTS=$3
SEED=$4
DEVICE=$5

# sh scripts/tritraining/base2novel_train.sh dtd 16 0 1 1
MODEL_DIR=output/${TRAINER}/base2novel_train/${DATASET}/shots_${SHOTS}/unlabeled_shots_${UNLABELED_SHOTS}/seed_${SEED}
DIR=output/${TRAINER}/base2novel_train/${DATASET}/shots_${SHOTS}/unlabeled_shots_${UNLABELED_SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=$DEVICE python tritrain_main_2.py \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --model-dir ${MODEL_DIR} \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.NUM_UNLABELED_SHOTS ${UNLABELED_SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.MODAL  base2novel \
    TRAIN_OR_TEST train 