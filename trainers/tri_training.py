import numpy as np
import sklearn
import pdb
from dassl.utils import save_checkpoint
import os.path as osp
import math
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
import pandas as pd
from torchvision.transforms import (
    Resize,
    Compose,
    ToTensor,
    Normalize,
    CenterCrop,
    RandomCrop,
    ColorJitter,
    RandomApply,
    GaussianBlur,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop
)
from dassl.utils import set_random_seed
from torchvision.transforms.functional import InterpolationMode


class Tri_Training:
    def __init__(self, cfg, base_estimator_1, base_estimator_2, base_estimator_3):
        self.cfg = cfg
        # 初始化函数，接受三个基本分类器
        self.estimators = [base_estimator_1, base_estimator_2, base_estimator_3]
        self.weak_augamentation, self.strong_augamentation = self.build_transform()
        self.min_new_label = -1  # wait to init

    def measure_error(self, datums, j, k):
        # 计算模型 j 和模型 k 之间的错误率
        print(f"Measuring error between model {j} and model {k}")
        y = [datum.label for datum in datums]
        # 获取模型 j 和 k 的预测结果
        j_pred = np.argmax(self.estimators[j].predict(datums), axis=1)
        k_pred = np.argmax(self.estimators[k].predict(datums), axis=1)

        print(f"Number of predictions: {len(y)}")

        # 获取两个模型预测相同且都预测错误的样本的 index
        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)

        print(
            f"Number of samples where model {j} and {k} are both wrong: {sum(wrong_index)}"
        )

        # 计算模型 j 和 k 预测一致的样本总数
        same_pred_count = sum(j_pred == k_pred)
        print(
            f"Number of samples where both models predict the same: {same_pred_count}"
        )

        # 避免除以零的情况，如果没有一致预测则返回 0
        if same_pred_count == 0:
            print("No samples where both models predict the same, returning 0")
            return 0

        # 计算两个模型在预测结果相同的情况下的错误率
        error_rate = sum(wrong_index) / same_pred_count
        print(
            f"The error rate of two models when the prediction results are the same: {error_rate}"
        )

        return error_rate

    def fit(self, train_x, train_u):
        num_classes = max(datum._real_label for datum in train_u) + 1
        self.min_new_label = math.ceil(num_classes / 2)
        warm_up_epochs = 5
        for i, model in enumerate(self.estimators):
            # 抽样
            sub_train_x = sklearn.utils.resample(train_x, replace=False, n_samples=len(train_x))
            print(f"------------Tritraining is fitting estimator: {i}------------")
            print(f"Training model {i} for {warm_up_epochs} epochs")
            set_random_seed(i + 1)
            model.fit(sub_train_x, max_epoch=warm_up_epochs)

        e_prime = [0.5] * 3
        l_prime = [0] * 3
        e = [0] * 3
        update = [False] * 3
        lb_train_u, lb_y = [[]] * 3, [[]] * 3
        improve = True
        iter = 0

        while improve:
            iter += 1
            print(f"------------------------iteration: {iter}------------------------")
            print(f"e_prime: {e_prime}")
            print(f"l_prime: {l_prime}")
            print(f"e: {e}")

            for i in range(3):
                print(f"----------------判断模型 {i} 是否需要更新----------------")
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False
                e[i] = self.measure_error(train_x, j, k)

                if e[i] < e_prime[i]:
                    # only mutual consistency
                    lb_train_u_mutual, lb_y_mutual = self.mutual_consistency(j, k, train_u)
                    # lb_train_u_self_j, lb_y_self_j = self.self_consistency(j, train_u)
                    # lb_train_u_self_k, lb_y_self_k = self.self_consistency(k, train_u)
                    # lb_train_u[i], lb_y[i] = self.merge_pseudo_labels(lb_train_u_mutual, lb_y_mutual, lb_train_u_self_j, lb_y_self_j, lb_train_u_self_k, lb_y_self_k)
                    lb_train_u_self_j, lb_y_self_j = self.self_consistency(j, lb_train_u_mutual, lb_y_mutual)
                    base_acc_j, new_acc_j = self.pseudo_label_acc(lb_train_u_self_j, lb_y_self_j)
                    print(f"1基类准确度为 {base_acc_j * 100:.2f}，1新类准确度为 {new_acc_j * 100:.2f}")
                    lb_train_u_self_k, lb_y_self_k = self.self_consistency(k, lb_train_u_mutual, lb_y_mutual)
                    base_acc_k, new_acc_k = self.pseudo_label_acc(lb_train_u_self_k, lb_y_self_k)
                    print(f"最终基类准确度为 {base_acc_k * 100:.2f}，最终新类准确度为 {new_acc_k * 100:.2f}")
                    if new_acc_k > new_acc_j:
                        lb_train_u[i], lb_y[i] = lb_train_u_self_k, lb_y_self_k 
                    else:
                        lb_train_u[i], lb_y[i] = lb_train_u_self_j, lb_y_self_j 

                    if l_prime[i] == 0:
                        l_prime[i] = int(e[i] ** 3 / (e_prime[i] ** 3 - e[i] ** 3) + 1)

                    print(f"e_prime: {e_prime}")
                    print(f"l_prime: {l_prime}")
                    print(f"e: {e}")

                    print(f"该轮可用伪标注数量: {len(lb_y[i])}")
                    print(f"前轮实际使用的伪标注数量: {l_prime[i]}")
                    # 该轮的伪标注数量大于前一轮的伪标注数量
                    if l_prime[i] < len(lb_y[i]):
                        print(f"该轮伪标注数量增加")
                        print(f"该轮估计的错误样本数量: {e[i] * len(lb_y[i])}")
                        print(f"前轮估计的错误样本数量: {e_prime[i] * l_prime[i]}")
                        # 错误样本数量减少即为满足更新条件
                        if e[i] * len(lb_y[i]) < e_prime[i] * l_prime[i]:
                            print(f"错误样本数量减少，更新模型 {i}")
                            update[i] = True
                        # 错误样本数量增加，但增加的数量不多
                        elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                            print(
                                f"错误样本数量增加, 但前轮伪标注数量大于 {e[i] / (e_prime[i] - e[i])}, 更新模型 {i}"
                            )
                            lb_index = np.random.choice(
                                len(lb_y[i]),
                                min(
                                    int((e_prime[i] / e[i]) * l_prime[i] - 1),
                                    len(lb_y[i]),
                                ),
                            )
                            lb_train_u[i] = [lb_train_u[i][idx] for idx in lb_index]
                            lb_y[i] = [lb_y[i][idx] for idx in lb_index]
                            update[i] = True

            # 更新每个模型（如果需要）
            for i in range(3):
                if update[i]:
                    print(f"----------------模型 {i} 正在被更新----------------")
                    # 将标记数据集与新标记的未标记样本合并，并重新训练模型
                    print(f"为模型 {i} 添加了 {len(lb_y[i])} 个新标记样本")
                    num_base_label = sum(1 for lb in lb_y[i] if lb < self.min_new_label)
                    num_new_label = sum(1 for lb in lb_y[i] if lb >= self.min_new_label)
                    print(f"划分到基类中的样本数量: {num_base_label}")
                    print(f"划分到新类中的样本数量: {num_new_label}")
                    self.estimators[i].fit(
                        labeled_datums=train_x,
                        unlabeled_datums=lb_train_u[i],
                        pseudo_labels=lb_y[i],
                        max_epoch=20,
                    )
                    # 更新 e_prime 和 l_prime
                    e_prime[i] = e[i]
                    l_prime[i] = len(lb_y[i])
                else:
                    print(f"----------------模型 {i} 无需被更新----------------")

            # 如果没有任何模型更新，结束循环
            if update == [False] * 3:
                improve = False
                print(f"TriTraining 阶段共迭代 {iter - 1} 个轮次")

        # 保存三个模型
        for estimator in self.estimators:
            estimator.custom_save_model()

        return

    def mutual_consistency(self, j, k, train_u):
        print(f"模型 {j} 预测中")
        # 使用未标记数据让模型 j 进行预测
        j_logits = self.estimators[j].predict(train_u)
        print(f"模型 {k} 预测中")
        # 使用未标记数据让模型 k 进行预测
        k_logits = self.estimators[k].predict(train_u)

        # -------------------------------
        # 不使用置信度过滤时的伪标签准确度统计
        # 直接取每个模型的最大概率预测作为伪标签
        ulb_y_j_old = np.argmax(j_logits, axis=1)
        ulb_y_k_old = np.argmax(k_logits, axis=1)

        # 统计两个模型预测一致的样本数量
        consistent_mask_old = ulb_y_j_old == ulb_y_k_old
        lb_train_u_old = [
            train_u[idx] for idx, is_true in enumerate(consistent_mask_old) if is_true
        ]
        lb_y_old = [
            ulb_y_j_old[idx]
            for idx, is_true in enumerate(consistent_mask_old)
            if is_true
        ]
        base_acc_old, new_acc_old = self.pseudo_label_acc(lb_train_u_old, lb_y_old)
        print(f"基于置信度筛选前，共有 {len(lb_train_u_old)} 个伪标注")
        print(
            f"基类准确度为 {base_acc_old * 100:.2f}，新类准确度为 {new_acc_old * 100:.2f}"
        )
        # -------------------------------

        ############################
        # add confidence bound
        ############################
        base_confidence_bound = 0.99
        new_confidence_bound = 0.9
        j_confidence = self.calculate_confidence(j_logits)
        k_confidence = self.calculate_confidence(k_logits)
        ulb_y_j = np.where(
            (
                (j_confidence > base_confidence_bound)
                & (np.argmax(j_logits, axis=1) < self.min_new_label)
            )
            | (
                (j_confidence > new_confidence_bound)
                & (np.argmax(j_logits, axis=1) >= self.min_new_label)
            ),
            np.argmax(j_logits, axis=1),
            -1,
        )
        ulb_y_k = np.where(
            (
                (k_confidence > base_confidence_bound)
                & (np.argmax(k_logits, axis=1) < self.min_new_label)
            )
            | (
                (k_confidence > new_confidence_bound)
                & (np.argmax(k_logits, axis=1) >= self.min_new_label)
            ),
            np.argmax(k_logits, axis=1),
            -1,
        )
        # 获取一致的伪标签
        consistent_mask = (ulb_y_j == ulb_y_k) & (ulb_y_j != -1)
        lb_train_u = [
            train_u[idx] for idx, is_true in enumerate(consistent_mask) if is_true
        ]
        lb_y = [ulb_y_j[idx] for idx, is_true in enumerate(consistent_mask) if is_true]
        # lb_y = np.array([train_u[idx]._real_label for idx, is_true in enumerate(consistent_mask) if is_true])        
        base_acc, new_acc = self.pseudo_label_acc(lb_train_u, lb_y, print_path=False)
        print(f"基于置信度筛选后，共有 {len(lb_train_u)} 个伪标注")
        print(f"基类准确度为 {base_acc * 100:.2f}，新类准确度为 {new_acc * 100:.2f}")
        print(f"伪标注数量减少了 {len(lb_train_u_old) - len(lb_train_u)}")
        print(
            f"基类准确度提高了 {(base_acc - base_acc_old) * 100:.2f}，新类准确度提高了 {(new_acc - new_acc_old)  * 100:.2f}"
        )
        return lb_train_u, lb_y

    def calculate_confidence(self, logits):
        """
        logits.shape: (num_samples, num_classes)
        """
        confidence = 1 - entropy(logits, axis=1) / np.log(logits.shape[1])
        return confidence

    def pseudo_label_acc(self, lb_train_u, lb_y, print_path=False):
        ############
        # 查看伪标签的精准度
        num_pseudo_base = sum(1 for lb in lb_y if lb < self.min_new_label)
        num_pseudo_new = sum(1 for lb in lb_y if lb >= self.min_new_label)
        real_labels = [datum._real_label for datum in lb_train_u]
        contrast = np.array(real_labels) == np.array(lb_y)
        base_acc = (
            sum(
                1
                for t in range(len(contrast))
                if contrast[t] and lb_y[t] < self.min_new_label
            )
            / num_pseudo_base if num_pseudo_base != 0 else 0
        )
        new_acc = (
            sum(
                1
                for t in range(len(contrast))
                if contrast[t] and lb_y[t] >= self.min_new_label
            )
            / num_pseudo_new if num_pseudo_new != 0 else 0
        )
        print(f"基类伪标注数量: {num_pseudo_base}")
        print(f"新类伪标注数量: {num_pseudo_new}")
        print(f"真实的基类数量: {sum(1 for lb in real_labels if lb < self.min_new_label)}")
        print(f"真实的新类数量: {sum(1 for lb in real_labels if lb >= self.min_new_label)}")
        if print_path:
            # 输出预测错误的样本
            paths = [datum.impath for datum in lb_train_u]  # 获取每个样本的路径
            wrong_paths = []
            for t in range(len(contrast)):
                if not contrast[t]:  # 如果预测错误
                    wrong_paths.append(paths[t])

            with open('paths.txt', 'a') as f:
                for path in wrong_paths:
                    f.write(path + '\n')
        
        
        return base_acc, new_acc
        #############

    # 合并自一致性和互一致性的伪标签集
    def merge_pseudo_labels(
        self,
        lb_train_u_mutual,
        lb_y_mutual,
        lb_train_u_self_j,
        lb_y_self_j,
        lb_train_u_self_k,
        lb_y_self_k,
    ):
        """
        合并互一致性和自一致性伪标签集：
        - 先去重；
        - 如果发生冲突，优先保留互一致性的结果。
        """
        # 初始化合并后的伪标签映射
        final_lb_train_u = []
        final_lb_y = []

        # 建立索引到标签的映射（互一致性优先）
        label_map = {}

        # Step 1: 添加互一致性标签（优先保留）
        for idx, sample in enumerate(lb_train_u_mutual):
            label_map[sample] = lb_y_mutual[idx]  # 直接存储互一致性结果

        # Step 2: 添加自一致性标签（如果存在冲突，忽略自一致性的结果）
        all_self_train_u = lb_train_u_self_j + lb_train_u_self_k
        all_self_y = lb_y_self_j + lb_y_self_k
        for idx, sample in enumerate(all_self_train_u):
            if sample not in label_map:  # 如果样本未被标记，直接添加
                label_map[sample] = all_self_y[idx]
            else:
                # 如果样本已存在，跳过（保留互一致性结果）
                pass

        # Step 3: 构建去重后的伪标签集
        for sample, label in label_map.items():
            final_lb_train_u.append(sample)
            final_lb_y.append(label)

        print(f"最终伪标签数量：{len(final_lb_train_u)}")
        # 输出最终伪标签的准确性
        base_acc, new_acc = self.pseudo_label_acc(final_lb_train_u, final_lb_y)
        print(f"最终基类准确度为 {base_acc * 100:.2f}，新类准确度为 {new_acc * 100:.2f}")
        return final_lb_train_u, final_lb_y

    def self_consistency(self, model_idx, train_u, label_u):

        print(f"模型 {model_idx} 自一致性检测中")
        # logits_no_aug = self.estimators[model_idx].predict(train_u)
        logits_weak = self.estimators[model_idx].predict(
            train_u, self.weak_augamentation
        )
        logits_strong = self.estimators[model_idx].predict(
            train_u, self.strong_augamentation
        )

        # 获取预测类别
        ulb_y_no_aug = label_u
        ulb_y_weak = np.argmax(logits_weak, axis=1)
        ulb_y_strong = np.argmax(logits_strong, axis=1)
        # mask1 = (ulb_y_no_aug == ulb_y_weak)
        # mask2 = (ulb_y_weak != ulb_y_strong)

        # # 统计每个条件为True的数量
        # num_true_mask1 = np.sum(mask1)
        # num_true_mask2 = np.sum(mask2)

        # print(f"条件1 (无增广 == 弱增广) 为True的数量: {num_true_mask1}")
        # print(f"条件2 (弱增广 != 强增广) 为True的数量: {num_true_mask2}")

        # 获取满足条件的伪标签（无增广 == 弱增广 != 强增广）
        consistent_mask = (ulb_y_no_aug == ulb_y_weak) & (ulb_y_weak != ulb_y_strong)
        lb_train_u = [
            train_u[idx] for idx, is_true in enumerate(consistent_mask) if is_true
        ]
        lb_y = [
            ulb_y_no_aug[idx] for idx, is_true in enumerate(consistent_mask) if is_true
        ]
        print(f"模型 {model_idx} 自一致性标记数量：{len(lb_train_u)}")
        print(f"其中无增广与弱增广预测一致但与强增广不一致的样本数：{np.sum(consistent_mask)}")
        return lb_train_u, lb_y

    def predict(self, datums):
        # 预测新数据集 X，返回预测结果
        print(f"Tritraining is predicting {len(datums)} samples")
        # 对三个模型分别进行预测，结果转换为 NumPy 数组
        output = [
            np.argmax(self.estimators[i].predict(datums), axis=1) for i in range(3)
        ]
        pred = np.asarray(output)

        # 如果模型 1 和 2 的预测一致，则以它们的预测结果为准
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]

        # 返回最终的预测结果
        y_pred = pred[0]
        return y_pred, output

    def build_transform(self):
        SIZE = (224, 224)
        PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
        PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
        interp_mode = InterpolationMode.BILINEAR
        input_size = SIZE

        print("Building weak augmentations")
        weak_aug = []

        # 添加 RandomHorizontalFlip
        print("+ random horizontal flip")
        weak_aug += [RandomHorizontalFlip()]

        # 确保图像大小足够再进行 RandomCrop
        print(
            f"+ resize the smaller edge to {max(input_size)} (ensure size > crop size)"
        )
        weak_aug += [Resize(max(input_size), interpolation=interp_mode)]

        print("+ random crop with padding=0.125 and padding_mode='reflect'")
        weak_aug += [
            RandomCrop(
                size=input_size,
                padding=int(0.125 * max(input_size)),
                padding_mode="reflect",
            )
        ]

        # 中心裁剪
        # print(f"+ {input_size[0]}x{input_size[1]} center crop")
        # weak_aug += [CenterCrop(input_size)]

        # 转为Tensor
        print("+ to torch tensor of range [0, 1]")
        weak_aug += [ToTensor()]

        # 归一化
        print(f"+ normalization (mean={PIXEL_MEAN}, std={PIXEL_STD})")
        weak_aug += [Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)]

        weak_aug = Compose(weak_aug)

        print("Building strong augmentations")
        # strong_aug = []
        # # 添加 RandomHorizontalFlip
        # print("+ random horizontal flip")
        # strong_aug += [RandomHorizontalFlip()]
        # # 确保图像大小足够再进行 RandomCrop
        # print(
        #     f"+ resize the smaller edge to {max(input_size)} (ensure size > crop size)"
        # )
        # strong_aug += [Resize(max(input_size), interpolation=interp_mode)]
        # print("+ random crop with padding=0.125 and padding_mode='reflect'")
        # strong_aug += [
        #     RandomCrop(
        #         size=input_size,
        #         padding=int(0.125 * max(input_size)),
        #         padding_mode="reflect",
        #     )
        # ]
        # # 添加随机颜色抖动
        # print("+ random color jitter")
        # strong_aug += [
        #     ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        # ]
        # # 随机灰度
        # print("+ random grayscale")
        # strong_aug += [RandomGrayscale(p=0.2)]
        # # 随机模糊
        # print("+ random gaussian blur")
        # strong_aug += [RandomApply([GaussianBlur(kernel_size=3)], p=0.1)]
        # # 中心裁剪
        # print(f"+ {input_size[0]}x{input_size[1]} center crop")
        # strong_aug += [CenterCrop(input_size)]

        # # 转为Tensor
        # print("+ to torch tensor of range [0, 1]")
        # strong_aug += [ToTensor()]

        # # 归一化
        # print(f"+ normalization (mean={PIXEL_MEAN}, std={PIXEL_STD})")
        # strong_aug += [Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)]

        # strong_aug = Compose(strong_aug)

        strong_aug = []
        cfg = self.cfg
        # 随机裁剪
        crop_padding = cfg.INPUT.CROP_PADDING
        print(f"+ random crop (padding = {crop_padding})")
        strong_aug += [RandomCrop(input_size, padding=crop_padding)]

        # 随机水平翻转
        print("+ random flip")
        strong_aug += [RandomHorizontalFlip()]

        # 随机色彩抖动
        b_ = cfg.INPUT.COLORJITTER_B
        c_ = cfg.INPUT.COLORJITTER_C
        s_ = cfg.INPUT.COLORJITTER_S
        h_ = cfg.INPUT.COLORJITTER_H
        print(
            f"+ color jitter (brightness={b_}, "
            f"contrast={c_}, saturation={s_}, hue={h_})"
        )
        strong_aug += [
            ColorJitter(
                brightness=b_,
                contrast=c_,
                saturation=s_,
                hue=h_,
            )
        ]

        # 随机高斯模糊
        gb_k, gb_p = cfg.INPUT.GB_K, cfg.INPUT.GB_P
        print(f"+ gaussian blur (kernel={gb_k}, p={gb_p})")
        strong_aug += [RandomApply([GaussianBlur(gb_k)], p=gb_p)]

        # 随机裁剪后调整大小
        s_ = cfg.INPUT.RRCROP_SCALE
        print(f"+ random resized crop (size={input_size}, scale={s_})")
        strong_aug += [
            RandomResizedCrop(input_size, scale=s_, interpolation=interp_mode)
        ]

        # 转换为张量
        print("+ to torch tensor of range [0, 1]")
        strong_aug += [ToTensor()]

        # 正则化
        print(
            f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})"
        )
        strong_aug += [Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)]

        strong_aug = Compose(strong_aug)

        return weak_aug, strong_aug
