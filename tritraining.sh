#!/bin/bash


for shot in 1 2 4 8 16
    do
        sh scripts/tritraining/base2novel_train.sh caltech101 $shot 0 3
        sh scripts/tritraining/base2novel_test.sh caltech101 $shot 0 3
    done
done

for dataset in dtd eurosat fgvc_aircraft oxford_flowers oxford_pets ucf101 food101 stanford_cars sun397
do 
    for seed in 1 2 3
    do 
        for shot in 1 2 4 8 16
        do
            sh scripts/tritraining/base2novel_train.sh $dataset $shot 0 $seed
            sh scripts/tritraining/base2novel_test.sh $dataset $shot 0 $seed
        done
    done
done