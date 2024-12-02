#!/bin/bash
# eurosat dtd oxford_flowers 
# for dataset in fgvc_aircraft stanford_cars 
# do 
#     for seed in 1 2 3
#     do 
#         sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed
#         sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed
#         sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed
#     done
# done

for dataset in caltech101
do 
    for seed in 3
    do 
        sh scripts/tritraining/base2novel_train_0.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_base_0.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_new_0.sh $dataset 16 0 $seed
    done
done


for dataset in sun397
do 
    for seed in 1 2 3
    do 
        sh scripts/tritraining/base2novel_train_0.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_base_0.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_new_0.sh $dataset 16 0 $seed
    done
done