DEVICE=3
sh scripts/tritraining/ssl_train_coop.sh cifar10 4 0 1 $DEVICE
sh scripts/tritraining/ssl_test_coop.sh cifar10 4 0 1 $DEVICE

sh scripts/tritraining/ssl_train_coop.sh cifar10 4 0 2 $DEVICE
sh scripts/tritraining/ssl_test_coop.sh cifar10 4 0 2 $DEVICE

sh scripts/tritraining/ssl_train_coop.sh cifar10 4 0 3 $DEVICE
sh scripts/tritraining/ssl_test_coop.sh cifar10 4 0 3 $DEVICE

sh scripts/tritraining/ssl_train.sh cifar10 25 0 1 $DEVICE
sh scripts/tritraining/ssl_test.sh cifar10 25 0 1 $DEVICE

sh scripts/tritraining/ssl_train_coop.sh cifar10 25 0 2 $DEVICE
sh scripts/tritraining/ssl_test_coop.sh cifar10 25 0 2 $DEVICE

sh scripts/tritraining/ssl_train_coop.sh cifar10 25 0 3 $DEVICE
sh scripts/tritraining/ssl_test_coop.sh cifar10 25 0 3 $DEVICE


sh scripts/tritraining/ssl_train_coop.sh cifar10 400 0 1 $DEVICE
sh scripts/tritraining/ssl_test_coop.sh cifar10 400 0 1 $DEVICE

sh scripts/tritraining/ssl_train_coop.sh cifar10 400 0 2 $DEVICE
qsh scripts/tritraining/ssl_test_coop.sh cifar10 400 0 2 $DEVICE

sh scripts/tritraining/ssl_train_coop.sh cifar10 400 0 3 $DEVICE
sh scripts/tritraining/ssl_test_coop.sh cifar10 400 0 3 $DEVICE
