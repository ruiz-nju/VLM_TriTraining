DATA=/mnt/hdd/zhurui/data
TRAINER=TriTraining

DATASET=$1
SHOTS=$2
UNLABELED_SHOTS=$3
SEED=$4

MODEL_DIR=1

DIR=output/${TRAINER}/base2novel_test/${DATASET}/shots_${SHOTS}/unlabeled_shots_${UNLABELED_SHOTS}/seed_${SEED}
CUDA_VISIBLE_DEVICES=1 python tritraining_main.py \
    --root ${DATA} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.SUBSAMPLE_CLASSES new \
    TRAINER.MODAL  base2novel