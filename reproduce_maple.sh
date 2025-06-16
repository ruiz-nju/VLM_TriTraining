DEVICE=0

# for dataset in imagenet
for dataset in caltech101 dtd eurosat oxford_pets stanford_cars fgvc_aircraft food101 oxford_flowers ucf101 sun397
do 
    for seed in 1 2 3
    do
        sh scripts/maple/base2novel_train.sh $dataset vit_b16 16 $seed $DEVICE
        sh scripts/maple/base2novel_test.sh $dataset vit_b16 16 $seed $DEVICE
    done
done
