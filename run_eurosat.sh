DEVICE=0
for dataset in eurosat
do 
    for seed in 1 2 3
    do 
        sh scripts/tritraining/base2novel_train.sh $dataset CoOp 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_base.sh $dataset CoOp 16 0 $seed $DEVICE 
        sh scripts/tritraining/base2novel_test_new.sh $dataset CoOp 16 0 $seed $DEVICE
    done
done
