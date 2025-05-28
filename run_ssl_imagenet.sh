DEVICE=0
sh scripts/tritraining/ssl_train.sh imagenet 4 0 1 $DEVICE
sh scripts/tritraining/ssl_test.sh imagenet 4 0 1 $DEVICE

sh scripts/tritraining/ssl_train.sh imagenet 4 0 2 $DEVICE
sh scripts/tritraining/ssl_test.sh imagenet 4 0 2 $DEVICE

sh scripts/tritraining/ssl_train.sh imagenet 4 0 3 $DEVICE
sh scripts/tritraining/ssl_test.sh imagenet 4 0 3 $DEVICE

sh scripts/tritraining/ssl_train.sh imagenet 25 0 1 $DEVICE
sh scripts/tritraining/ssl_test.sh imagenet 25 0 1 $DEVICE

sh scripts/tritraining/ssl_train.sh imagenet 25 0 2 $DEVICE
sh scripts/tritraining/ssl_test.sh imagenet 25 0 2 $DEVICE

sh scripts/tritraining/ssl_train.sh imagenet 25 0 3 $DEVICE
sh scripts/tritraining/ssl_test.sh imagenet 25 0 3 $DEVICE


sh scripts/tritraining/ssl_train.sh imagenet 100 0 1 $DEVICE
sh scripts/tritraining/ssl_test.sh imagenet 100 0 1 $DEVICE

sh scripts/tritraining/ssl_train.sh imagenet 100 0 2 $DEVICE
sh scripts/tritraining/ssl_test.sh imagenet 100 0 2 $DEVICE

sh scripts/tritraining/ssl_train.sh imagenet 100 0 3 $DEVICE
sh scripts/tritraining/ssl_test.sh imagenet 100 0 3 $DEVICE
