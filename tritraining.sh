#!/bin/bash


# for shot in 1 2 4 8 16
#     do
#         sh scripts/tritraining/base2novel_train.sh caltech101 $shot 0 3
#         sh scripts/tritraining/base2novel_test.sh caltech101 $shot 0 3
#     done
# done

for dataset in dtd fgvc_aircraft caltech101
do 
    for seed in 1
    do 
        sh scripts/tritraining/base2novel_train.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_base.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_new.sh $dataset 16 0 $seed
    done
done