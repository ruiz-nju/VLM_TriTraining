import numpy as np
import sklearn
import pdb
from dassl.utils import save_checkpoint
import os.path as osp
import math
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
import pandas as pd


class Tri_Training:
    def __init__(self, base_estimator_1, base_estimator_2, base_estimator_3):
        # 初始化函数，接受三个基本分类器
        self.estimators = [base_estimator_1, base_estimator_2, base_estimator_3]

    def measure_error(self, datums, j, k):
        # 计算模型 j 和模型 k 之间的错误率
        print(f"Measuring error between model {j} and model {k}")
        y = [datum.label for datum in datums]
        # 获取模型 j 和 k 的预测结果
        j_pred = np.argmax(self.estimators[j].predict(datums), axis=1)
        k_pred = np.argmax(self.estimators[k].predict(datums), axis=1)

        # 打印模型 j 和模型 k 的前 10 个预测结果，方便调试
        print(f"Number of predictions: {len(y)}")

        # 获取两个模型都预测错误的样本的 index
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
        min_new_label = math.ceil(num_classes / 2)
        # 初始化每个分类器的训练，使用带放回抽样生成新的训练集
        for i, model in enumerate(self.estimators):
            sub_train_x = sklearn.utils.resample(train_x)
            print(f"------------Tritraining is fitting estimator: {i}------------")
            # model.fit(sub_train_x)
            # # 保存模型
            # model.custom_save_model(temp_dir="pretraining_50epoch")
            load_dir = osp.join(
                "pretraining_50epoch",
                model.cfg.OUTPUT_DIR,
                model.cfg.TRAINER.NAME,
            )
            model.custom_load_model(load_dir)

        # e_prime: 用于存储每个模型的初始错误率，初始化为 0.5
        e_prime = [0.5] * 3
        # l_prime: 用于记录每个模型的标记样本数量
        l_prime = [0] * 3
        # e: 用于存储每次迭代计算的错误率
        e = [0] * 3
        # update: 标记是否需要更新模型
        update = [False] * 3
        # lb_X 和 lb_y: 用于存储标记样本的特征和标签
        lb_train_u, lb_y = [[]] * 3, [[]] * 3
        improve = True
        iter = 0

        # 现在只迭代一个轮次，因为之前测试迭代两个轮次后性能反而下降了
        # while improve and iter < 1:
        while improve:
            iter += 1
            print("iteration:", iter)
            print(f"e_prime: {e_prime}")
            print(f"l_prime: {l_prime}")
            print(f"e: {e}")

            # 对每个模型 i 进行错误率计算并决定是否更新
            for i in range(3):
                print(f"----------------判断模型 {i} 是否需要更新----------------")
                # j 和 k 是除 i 以外的两个模型
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False

                # 计算模型 j 和 k 的相对错误率
                e[i] = self.measure_error(train_x, j, k)

                if e[i] < e_prime[i]:
                    base_confidence_bound = 0.9
                    new_confidence_bound = 0.7
                    print(f"模型 {j} 预测中")
                    # 使用未标记数据让模型 j 进行预测
                    j_logits = self.estimators[j].predict(train_u)
                    ulb_y_j = np.argmax(j_logits, axis=1)
                    # j_confidence = self.calculate_confidence(j_logits)
                    # # 置信度低的样本使用 -1 标记
                    # ulb_y_j = np.where(
                    #     (
                    #         (j_confidence > base_confidence_bound)
                    #         & (np.argmax(j_logits, axis=1) < min_new_label)
                    #     )
                    #     | (
                    #         (j_confidence > new_confidence_bound)
                    #         & (np.argmax(j_logits, axis=1) >= min_new_label)
                    #     ),
                    #     np.argmax(j_logits, axis=1),
                    #     -1,
                    # )
                    print(f"模型 {k} 预测中")
                    # 使用未标记数据让模型 k 进行预测
                    k_logits = self.estimators[k].predict(train_u)
                    ulb_y_k = np.argmax(k_logits, axis=1)
                    # k_confidence = self.calculate_confidence(k_logits)
                    # ulb_y_k = np.where(
                    #     (
                    #         (k_confidence > base_confidence_bound)
                    #         & (np.argmax(k_logits, axis=1) < min_new_label)
                    #     )
                    #     | (
                    #         (k_confidence > new_confidence_bound)
                    #         & (np.argmax(k_logits, axis=1) >= min_new_label)
                    #     ),
                    #     np.argmax(k_logits, axis=1),
                    #     -1,
                    # )

                    # 获取 j 和 k 预测一致且均不为 - 1 的未标记样本，并将它们作为模型 i 的新标记样本
                    consistent_mask = (ulb_y_j == ulb_y_k) & (ulb_y_j != -1)
                    lb_train_u[i] = [
                        train_u[idx]
                        for idx, is_true in enumerate(consistent_mask)
                        if is_true
                    ]
                    lb_y[i] = [
                        ulb_y_j[idx]
                        for idx, is_true in enumerate(consistent_mask)
                        if is_true
                    ]

                    ############
                    # 查看伪标签的精准度
                    num_pseudo_base = sum(1 for lb in lb_y[i] if lb < min_new_label)
                    num_pseudo_new = sum(1 for lb in lb_y[i] if lb >= min_new_label)
                    real_labels = [datum._real_label for datum in lb_train_u[i]]
                    contrast = np.array(real_labels) == np.array(lb_y[i])
                    print(
                        f"基类伪标注的精确度: {sum(1 for t in range(len(contrast)) if contrast[t] and lb_y[i][t] < min_new_label) / num_pseudo_base}"
                    )
                    print(
                        f"新类伪标注的精确度: {sum(1 for t in range(len(contrast)) if contrast[t] and lb_y[i][t] >= min_new_label) / num_pseudo_new}"
                    )
                    #############

                    if l_prime[i] == 0:
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)

                    print(f"e_prime: {e_prime}")
                    print(f"l_prime: {l_prime}")
                    print(f"e: {e}")

                    print(f"该轮伪标注数量: {len(lb_y[i])}")
                    print(f"前轮伪标注数量: {l_prime[i]}")
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
                            # 随机选择部分样本来更新模型
                            # lb_index = np.random.choice(len(lb_y[i]), len(lb_y[i]) // 2)
                            lb_index = np.random.choice(
                                len(lb_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1)
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
                    num_base_label = sum(1 for lb in lb_y[i] if lb < min_new_label)
                    num_new_label = sum(1 for lb in lb_y[i] if lb >= min_new_label)
                    print(f"划分到基类中的样本数量: {num_base_label}")
                    print(f"划分到新类中的样本数量: {num_new_label}")
                    self.estimators[i].fit(
                        train_x, lb_train_u[i], lb_y[i], max_epoch=25
                    )
                    # 更新 e_prime 和 l_prime
                    e_prime[i] = e[i]
                    l_prime[i] = len(lb_y[i])

            # 如果没有任何模型更新，结束循环
            if update == [False] * 3:
                improve = False
                print(f"TriTraining 阶段共迭代 {iter - 1} 个轮次")

        # 保存三个模型
        for estimator in self.estimators:
            estimator.custom_save_model()

        return

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

    def calculate_confidence(self, logits):
        """
        logits.shape: (num_samples, num_classes)
        """
        confidence = 1 - entropy(logits, axis=1) / np.log(logits.shape[1])
        return confidence
