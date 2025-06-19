# 用于自动计算 HM 等数据
import numpy as np
import os.path as osp
import warnings
from dassl.utils import check_isfile
import sys

# 禁用所有警告
warnings.filterwarnings("ignore")


def hm(a, b):
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def read_result(file: str):
    try:
        with open(file, "r") as f:
            lines = f.readlines()

        target_lines = lines[-5:-1]
        # 提取 ":" 后面的数值
        # 形式为 * accuracy: 60.1%
        for target_line in target_lines:
            if "accuracy" in target_line:
                return float(target_line.split(":")[-1].strip().replace("%", ""))
    except Exception:
        return None  # 直接返回 None，跳过报错


def main():
    classifier = "CoOp"
    datasets = [
        "cifar10",
        "cifar100",
        "imagenet100",
    ]
    data_dir = "output_openworld"
    for dataset in datasets:
        base_accs = []
        novel_accs = []
        all_accs = []
        for seed in range(1, 4):
            base_file = osp.join(data_dir, classifier, "test_base", dataset, "shots_16", classifier, "vit_b16_ep50", f"seed_{seed}", "log.txt")
            novel_file = osp.join(data_dir, classifier, "test_new", dataset, "shots_16", classifier, "vit_b16_ep50", f"seed_{seed}", "log.txt")
            all_file = osp.join(data_dir, classifier, "test_all", dataset, "shots_16", classifier, "vit_b16_ep50", f"seed_{seed}", "log.txt")
            if check_isfile(base_file):
                base_accs.append(read_result(base_file))
            if check_isfile(novel_file):
                novel_accs.append(read_result(novel_file))
            if check_isfile(all_file):
                all_accs.append(read_result(all_file))
        print(dataset)
        # 输出均值和方差，形式为 a±b
        print(f"base: {np.mean(base_accs):.2f}±{np.std(base_accs):.2f}")
        print(f"novel: {np.mean(novel_accs):.2f}±{np.std(novel_accs):.2f}")
        print(f"all: {np.mean(all_accs):.2f}±{np.std(all_accs):.2f}")

if __name__ == "__main__":
    main()
