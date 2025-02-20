# 用于自动计算 HM 等数据
import numpy as np
import os.path as osp
import warnings
from dassl.utils import check_isfile

# 禁用所有警告
warnings.filterwarnings("ignore")


def hm(a, b):
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def read_result(file: str):
    try:
        with open(file, "r") as f:
            lines = f.readlines()

        result = []
        target_lines = [lines[-4], lines[-3], lines[-2], lines[-1]]

        for target_line in target_lines:
            # 提取 ":" 后面的数值
            if "Accuracy" in target_line:
                result.append(float(target_line.split(":")[-1].strip()))
        return result
    except Exception:
        return None  # 直接返回 None，跳过报错


def show_result(dataset: str):
    dir_name = "output"
    base_dir = osp.join(
        dir_name,
        "TriTraining/base2novel_test_base",
        dataset,
        "shots_16/unlabeled_shots_0",
    )
    new_dir = osp.join(
        dir_name,
        "TriTraining/base2novel_test_new",
        dataset,
        "shots_16/unlabeled_shots_0",
    )

    base_accs = []
    new_accs = []
    hm_accs = []

    for seed in range(1, 4):
        try:
            # 构建文件路径
            base_file = osp.join(base_dir, f"seed_{seed}", "log.txt")
            new_file = osp.join(new_dir, f"seed_{seed}", "log.txt")

            if check_isfile(base_file) and check_isfile(new_file):
                # 读取 Accuracy 并计算 HM
                base_acc = read_result(base_file)
                new_acc = read_result(new_file)

                if base_acc is not None and new_acc is not None:
                    hm_acc = hm(base_acc[0], new_acc[0])
                    base_accs.append(base_acc[0])
                    new_accs.append(new_acc[0])
                    hm_accs.append(hm_acc)

                    # 打印单个 seed 的结果
                    print(
                        # f"Seed {seed}: base: {base_acc[0] * 100:.2f} ({base_acc[1] * 100:.2f} {base_acc[2] * 100:.2f} {base_acc[3] * 100:.2f}), "
                        # f"new: {new_acc[0] * 100:.2f} ({new_acc[1] * 100:.2f} {new_acc[2] * 100:.2f} {new_acc[3] * 100:.2f}), "
                        # f"hm: {hm_acc * 100:.2f}"
                        f"Seed {seed}: base: {base_acc[0] * 100:.2f}, new: {new_acc[0] * 100:.2f}, hm: {hm_acc * 100:.2f}"
                    )
        except Exception:
            # 静默忽略任何异常
            continue

    # 如果所有种子的结果都成功读取，计算平均值
    if len(base_accs) == 3 and len(new_accs) == 3 and len(hm_accs) == 3:
        avg_base = np.mean(base_accs)
        avg_new = np.mean(new_accs)
        avg_hm = np.mean(hm_accs)
        print(
            f"Dataset {dataset} Average: base: {avg_base * 100:.2f}, new: {avg_new * 100:.2f}, hm: {avg_hm * 100:.2f}"
        )
        return avg_base, avg_new, avg_hm
    else:
        return None, None, None


def main():
    datasets = [
        "dtd",
        "eurosat",
        "fgvc_aircraft",
        "oxford_flowers",
        "caltech101",
        "food101",
        "oxford_pets",
        "ucf101",
        "sun397",
        # "stanford_cars",
    ]

    all_avg_base = []
    all_avg_new = []
    all_avg_hm = []

    for dataset in datasets:
        try:
            print(f"---- Dataset: {dataset} ----")
            avg_base, avg_new, avg_hm = show_result(dataset)
            if avg_base is not None and avg_new is not None and avg_hm is not None:
                all_avg_base.append(avg_base)
                all_avg_new.append(avg_new)
                all_avg_hm.append(avg_hm)
        except Exception:
            # 静默忽略任何异常
            continue

    # 如果所有数据集都有平均值，计算整体平均值
    if len(all_avg_base) == len(datasets):
        overall_avg_base = np.mean(all_avg_base)
        overall_avg_new = np.mean(all_avg_new)
        overall_avg_hm = np.mean(all_avg_hm)
        print(
            f"Overall Average: base: {overall_avg_base * 100:.2f}, new: {overall_avg_new * 100:.2f}, hm: {overall_avg_hm * 100:.2f}"
        )


if __name__ == "__main__":
    main()
