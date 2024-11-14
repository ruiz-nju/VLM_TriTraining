#!/bin/bash

# for dataset in  dtd fgvc_aircraft eurosat oxford_flowers
# do 
#     for seed in 1
#     do 
#         sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed
#         sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed
#         sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed
#     done
# done

for dataset in sun397 
do 
    for seed in 1
    do 
        sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed
    done
done

# for dataset in sun397 caltech101 food101 oxford_pets stanford_cars  ucf101
# do 
#     for seed in 1
#     do 
#         sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed
#         sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed
#         sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed
#     done
# done