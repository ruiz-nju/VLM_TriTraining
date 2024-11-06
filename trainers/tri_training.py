import numpy as np
import sklearn
import pdb
from dassl.utils import save_checkpoint
import os.path as osp
import math


class Tri_Training:
    def __init__(self, base_estimator_1, base_estimator_2, base_estimator_3):
        # 初始化函数，接受三个基本分类器
        self.estimators = [base_estimator_1, base_estimator_2, base_estimator_3]

    def measure_error(self, datums, j, k):
        # 计算模型 j 和模型 k 之间的错误率
        print(f"Measuring error between model {j} and model {k}")
        y = [datum.label for datum in datums]
        # 获取模型 j 和 k 的预测结果
        j_pred = self.estimators[j].predict(datums)
        k_pred = self.estimators[k].predict(datums)

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
        # print(len(train_x), len(train_u)) # 800 800
        # print(train_x[0]) # <dassl.data.datasets.base_dataset.Datum object at 0x75ecd4b667c0>
        # print(train_u[0]) # <dassl.data.datasets.base_dataset.Datum object at 0x75ecd4cb35e0>
        num_classes = max(datum._real_label for datum in train_u) + 1
        min_new_label = math.ceil(num_classes / 2)
        # 初始化每个分类器的训练，使用带放回抽样生成新的训练集
        for i in range(3):
            sub_train_x = sklearn.utils.resample(train_x)
            print(f"------------Tritraining is fitting estimator: {i}------------")
            self.estimators[i].fit(sub_train_x)

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

        while improve and iter < 1:
            iter += 1
            print("iteration:", iter)

            # 对每个模型 i 进行错误率计算并决定是否更新
            for i in range(3):
                # j 和 k 是除 i 以外的两个模型
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False

                # 计算模型 j 和 k 的相对错误率
                e[i] = self.measure_error(train_x, j, k)

                # 如果新的错误率小于先前的错误率，尝试用未标记数据更新 i
                if e[i] < e_prime[i]:
                    print(f"----------------{j} is predicting----------------")
                    # 使用未标记数据让模型 j 进行预测
                    ulb_y_j = self.estimators[j].predict(train_u)
                    print(f"----------------{k} is predicting----------------")
                    # 使用未标记数据让模型 k 进行预测
                    ulb_y_k = self.estimators[k].predict(train_u)

                    # 获取 j 和 k 预测一致的未标记样本，并将它们作为模型 i 的新标记样本
                    mask = (ulb_y_j == ulb_y_k).tolist()
                    lb_train_u[i] = [
                        train_u[idx] for idx, is_true in enumerate(mask) if is_true
                    ]
                    lb_y[i] = [
                        ulb_y_j[idx] for idx, is_true in enumerate(mask) if is_true
                    ]

                    # 更新 l_prime 为所需标记样本的数量
                    if l_prime[i] == 0:
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)

                    # 如果满足更新条件，则更新标记数据集
                    if l_prime[i] < len(lb_y[i]):
                        # 错误样本数量减少即为满足更新条件
                        if e[i] * len(lb_y[i]) < e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                            # 随机选择部分样本来更新模型
                            lb_index = np.random.choice(len(lb_y[i]), len(lb_y[i]) // 2)
                            lb_train_u[i] = [lb_train_u[i][idx] for idx in lb_index]
                            lb_y[i] = [lb_y[i][idx] for idx in lb_index]
                            update[i] = True

            # 更新每个模型（如果需要）
            for i in range(3):
                if update[i]:
                    print(f"----------------{i} is being updated----------------")
                    # 将标记数据集与新标记的未标记样本合并，并重新训练模型
                    print(f"Add {len(lb_y[i])} new labeled samples to model {i}")
                    num_base_label = sum(1 for lb in lb_y[i] if lb < min_new_label)
                    num_new_label = sum(1 for lb in lb_y[i] if lb >= min_new_label)
                    print(f"Number of base labels: {num_base_label}")
                    print(f"Number of new labels: {num_new_label}")
                    self.estimators[i].fit(
                        train_x, lb_train_u[i], lb_y[i], max_epoch=20
                    )
                    # self.estimators[i].fit(train_x, lb_train_u[i], lb_y[i], max_epoch=2)
                    # 更新 e_prime 和 l_prime
                    e_prime[i] = e[i]
                    l_prime[i] = len(lb_y[i])

            # 如果没有任何模型更新，结束循环
            if update == [False] * 3:
                improve = False

        # 保存三个模型
        for estimator in self.estimators:
            self.save_model(estimator)

        return

    def predict(self, datums):
        # 预测新数据集 X，返回预测结果
        print(f"Tritraining is predicting {len(datums)} samples")
        # 对三个模型分别进行预测，结果转换为 NumPy 数组
        output = [self.estimators[i].predict(datums).cpu().numpy() for i in range(3)]
        pred = np.asarray(output)

        # 如果模型 1 和 2 的预测一致，则以它们的预测结果为准
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]

        # 返回最终的预测结果
        y_pred = pred[0]
        return y_pred

    def save_model(self, estimator):
        names = estimator.get_model_names()
        for name in names:
            model_dict = estimator._models[name].state_dict()

            optim_dict = None
            if estimator._optims[name] is not None:
                optim_dict = estimator._optims[name].state_dict()

            sched_dict = None
            if estimator._scheds[name] is not None:
                sched_dict = estimator._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(estimator.output_dir, estimator.cfg.TRAINER.NAME, name),
                model_name="final_model.pth.tar",
                with_epoch=False,
            )
