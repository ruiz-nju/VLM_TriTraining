#!/bin/bash
DEVICE=0
# Flowers102, FGVCAircraft, DTD and EuroSAT
for dataset in oxford_flowers dtd
do 
    for seed in 1
    do 
        sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed $DEVICE
    done
done
# for dataset in ucf101
# do 
#     for seed in 2
#     do 
#         sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed $DEVICE
#         sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed $DEVICE
#         sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed $DEVICE
#     done
# done