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


def main(data_dir):
    accs = []
    for seed in range(1, 4):
        file = osp.join(data_dir, f"seed_{seed}", "log.txt")
        if check_isfile(file):
            accs.append(read_result(file))
    # 输出均值和方差，形式为 a±b
    print(accs)
    print(f"{np.mean(accs):.2f}±{np.std(accs):.2f}")


if __name__ == "__main__":
    # 从命令行读取数据集路径
    data_dir = sys.argv[1]
    main(data_dir)
