DATA=/data0/zhur/data
TRAINER=TriTraining

DATASET=$1
CLASSIFIER=$2
SHOTS=$3
UNLABELED_SHOTS=$4
SEED=$5
DEVICE=$6

MODEL_DIR=output/${TRAINER}/base2novel_train/${DATASET}/${CLASSIFIER}/shots_${SHOTS}/unlabeled_shots_${UNLABELED_SHOTS}/seed_${SEED}
DIR=output/${TRAINER}/base2novel_test_base/${DATASET}/${CLASSIFIER}/shots_${SHOTS}/unlabeled_shots_${UNLABELED_SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=$DEVICE python tritraining_main.py \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --classifier ${CLASSIFIER} \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.NUM_UNLABELED_SHOTS ${UNLABELED_SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.MODAL  base2novel \
    TRAIN_OR_TEST test