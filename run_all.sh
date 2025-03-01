#!/bin/bash

# 执行 tritraining0.sh
./tritraining0.sh &

# 执行 tritraining1.sh
./tritraining1.sh &

# 执行 tritraining2.sh
./tritraining2.sh &

# 执行 tritraining3.sh
./tritraining3.sh &

# 等待所有后台进程完成
wait

