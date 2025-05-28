DEVICE=1
sh scripts/tritraining/ssl_train.sh stl10 4 0 1 $DEVICE
sh scripts/tritraining/ssl_test.sh stl10 4 0 1 $DEVICE

sh scripts/tritraining/ssl_train.sh stl10 4 0 2 $DEVICE
sh scripts/tritraining/ssl_test.sh stl10 4 0 2 $DEVICE

sh scripts/tritraining/ssl_train.sh stl10 4 0 3 $DEVICE
sh scripts/tritraining/ssl_test.sh stl10 4 0 3 $DEVICE

sh scripts/tritraining/ssl_train.sh stl10 25 0 1 $DEVICE
sh scripts/tritraining/ssl_test.sh stl10 25 0 1 $DEVICE

sh scripts/tritraining/ssl_train.sh stl10 25 0 2 $DEVICE
sh scripts/tritraining/ssl_test.sh stl10 25 0 2 $DEVICE

sh scripts/tritraining/ssl_train.sh stl10 25 0 3 $DEVICE
sh scripts/tritraining/ssl_test.sh stl10 25 0 3 $DEVICE


sh scripts/tritraining/ssl_train.sh stl10 100 0 1 $DEVICE
sh scripts/tritraining/ssl_test.sh stl10 100 0 1 $DEVICE

sh scripts/tritraining/ssl_train.sh stl10 100 0 2 $DEVICE
sh scripts/tritraining/ssl_test.sh stl10 100 0 2 $DEVICE

sh scripts/tritraining/ssl_train.sh stl10 100 0 3 $DEVICE
sh scripts/tritraining/ssl_test.sh stl10 100 0 3 $DEVICE
