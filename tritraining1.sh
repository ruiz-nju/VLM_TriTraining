for dataset in ucf101
do 
    for seed in 1 2 3
    do 
        sh scripts/tritraining/base2novel_train_1.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_base_1.sh $dataset 16 0 $seed
        sh scripts/tritraining/base2novel_test_new_1.sh $dataset 16 0 $seed
    done
done