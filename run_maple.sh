DEVICE=1

for dataset in caltech101 dtd eurosat oxford_pets stanford_cars fgvc_aircraft food101 oxford_flowers ucf101 sun397 imagenet
do 
    for seed in 1 2 3
    do
        sh scripts/tritraining/base2novel_train.sh $dataset MaPLe 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_base.sh $dataset MaPLe 16 0 $seed $DEVICE
        sh scripts/tritraining/base2novel_test_new.sh $dataset MaPLe 16 0 $seed $DEVICE
    done
done
