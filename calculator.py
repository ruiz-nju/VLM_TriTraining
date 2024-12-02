# 用于自动 计算 HM 等数据


import numpy as np
import os.path as osp
from dassl.utils import check_isfile, listdir_nohidden


def hm(a, b):
    return 2 * a * b / (a + b)


def read_result(file: str):
    with open(file, "r") as f:
        lines = f.readlines()

    # 获取倒数第四行
    target_line = lines[-4]

    # 提取 "Accuracy" 后面的数值
    if "Accuracy" in target_line:
        return float(target_line.split(":")[-1].strip())
    else:
        raise ValueError(f"Accuracy not found in {file}")


def parse(dataset: str):
    base_dir = osp.join(
        "output/TriTraining/base2novel_test_base",
        dataset,
        "shots_16/unlabeled_shots_0",
    )
    new_dir = osp.join(
        "output/TriTraining/base2novel_test_new",
        dataset,
        "shots_16/unlabeled_shots_0",
    )
    result = [0.0, 0.0, 0.0]

    for seed in range(1, 4):
        # 构建文件路径
        base_file = osp.join(base_dir, f"seed_{seed}", "log.txt")
        new_file = osp.join(new_dir, f"seed_{seed}", "log.txt")

        # 确保文件存在
        assert check_isfile(base_file), f"File not found: {base_file}"
        assert check_isfile(new_file), f"File not found: {new_file}"

        # 读取 Accuracy 并计算 HM
        base_acc = read_result(base_file)
        new_acc = read_result(new_file)
        hm_acc = hm(base_acc, new_acc)

        # 打印单个 seed 的结果
        print(
            f"Seed {seed}: base: {base_acc * 100:.2f}, new: {new_acc * 100:.2f}, hm: {hm_acc * 100:.2f}"
        )

        # 累加结果
        result[0] += base_acc
        result[1] += new_acc
        result[2] += hm_acc

    # 计算平均值
    result = [r / 3 for r in result]
    print(
        f"Average: base: {result[0] * 100:.2f}, new: {result[1] * 100:.2f}, hm: {result[2] * 100:.2f}"
    )
    return result


def main():
    datasets = [
        "stanford_cars",
        "oxford_flowers",
        "fgvc_aircraft",
        "dtd",
        "eurosat",
        # "caltech101",
        "food101",
        "oxford_pets",
        # "sun397",
        # "ucf101",
    ]

    total_result = [0.0, 0.0, 0.0]  # 用于累加所有数据集的结果

    for dataset in datasets:
        print(f"---- Dataset: {dataset} ----")
        result = parse(dataset)
        # 累加每个数据集的结果
        total_result = [total_result[i] + result[i] for i in range(3)]

    # 计算所有数据集的平均值
    total_result = [r / len(datasets) for r in total_result]
    print(f"---- Overall Results ----")
    print(
        f"Average: base: {total_result[0] * 100:.2f}, new: {total_result[1] * 100:.2f}, hm: {total_result[2] * 100:.2f}"
    )


if __name__ == "__main__":
    main()
