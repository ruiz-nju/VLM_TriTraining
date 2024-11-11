#!/bin/bash


# for dataset in  fgvc_aircraft dtd eurosat oxford_flowers
# do 
#     for seed in 1
#     do 
#         sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed
#         sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed
#         sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed
#     done
# done

for dataset in caltech101 food101 oxford_pets stanford_cars sun397 ucf101
do 
    for seed in 1
    do 
        # sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed
    done
done