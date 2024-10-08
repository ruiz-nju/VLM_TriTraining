import numpy as np
import sklearn
import pdb


class Tri_Training:
    def __init__(self, base_estimator_1, base_estimator_2, base_estimator_3):
        # 初始化函数，接受三个基本分类器
        self.estimators = [base_estimator_1, base_estimator_2, base_estimator_3]

    def measure_error(self, X, y, j, k):
        # 计算模型 j 和模型 k 之间的错误率
        print(f"Measuring error between model {j} and model {k}")

        # 获取模型 j 和 k 的预测结果
        j_pred = self.estimators[j].predict(X)
        k_pred = self.estimators[k].predict(X)

        # 打印模型 j 和模型 k 的前 10 个预测结果，方便调试
        print(f"Model {j} predictions (first 10): {j_pred[:10]}")
        print(f"Model {k} predictions (first 10): {k_pred[:10]}")

        # 获取模型 j 预测错误但 k 与 j 预测一致的样本索引
        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)

        # 打印模型 j 错误但 k 同意的样本数量和比例
        print(
            f"Number of samples where model {j} is wrong but {k} agrees: {sum(wrong_index)}"
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

        # 计算并返回模型 j 出错但 k 同意的比例（错误率）
        error_rate = sum(wrong_index) / same_pred_count
        print(f"Error rate for model {j} when both models agree: {error_rate}")

        return error_rate

    def fit(self, X, y, unlabeled_X):
        # 初始化每个分类器的训练，使用带放回抽样生成新的训练集
        for i in range(3):
            sub_X, sub_y = sklearn.utils.resample(X, y)
            print("fitting estimator:", i)
            self.estimators[i].fit(sub_X, sub_y)

        # e_prime: 用于存储每个模型的初始错误率，初始化为 0.5
        e_prime = [0.5] * 3
        # l_prime: 用于记录每个模型的标记样本数量
        l_prime = [0] * 3
        # e: 用于存储每次迭代计算的错误率
        e = [0] * 3
        # update: 标记是否需要更新模型
        update = [False] * 3
        # lb_X 和 lb_y: 用于存储标记样本的特征和标签
        lb_X, lb_y = [[]] * 3, [[]] * 3
        improve = True
        iter = 0

        # 不断迭代直到模型不再更新（即 update 全为 False）
        while improve:
            iter += 1
            print("iteration:", iter)

            # 对每个模型 i 进行错误率计算并决定是否更新
            for i in range(3):
                # j 和 k 是除 i 以外的两个模型
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False

                # 计算模型 j 和 k 对模型 i 的错误率
                e[i] = self.measure_error(X, y, j, k)

                # 如果模型 i 的错误率小于其先前的错误率，尝试用未标记数据更新它
                if e[i] < e_prime[i]:
                    print("predicting:", j)
                    # 使用未标记数据让模型 j 进行预测
                    ulb_y_j = self.estimators[j].predict(unlabeled_X)
                    print("predicting:", k)
                    # 使用未标记数据让模型 k 进行预测
                    ulb_y_k = self.estimators[k].predict(unlabeled_X)

                    # 获取 j 和 k 预测一致的未标记样本，并将它们作为模型 i 的新标记样本
                    lb_X[i] = unlabeled_X[ulb_y_j == ulb_y_k]
                    lb_y[i] = ulb_y_j[ulb_y_j == ulb_y_k]

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
                            lb_index = np.random.choice(
                                len(lb_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1)
                            )
                            lb_X[i], lb_y[i] = lb_X[i][lb_index], lb_y[i][lb_index]
                            update[i] = True

            # 更新每个模型（如果需要）
            for i in range(3):
                if update[i]:
                    print("updating estimator:", i)
                    # 将标记数据集与新标记的未标记样本合并，并重新训练模型
                    self.estimators[i].fit(
                        np.append(X, lb_X[i], axis=0), np.append(y, lb_y[i], axis=0)
                    )
                    # 更新 e_prime 和 l_prime
                    e_prime[i] = e[i]
                    l_prime[i] = len(lb_y[i])

            # 如果没有任何模型更新，结束循环
            if update == [False] * 3:
                improve = False

        return

    def predict(self, X):
        # 预测新数据集 X，返回预测结果
        print(f"predicting {len(X)} samples")
        # 对三个模型分别进行预测，结果转换为 NumPy 数组
        output = [self.estimators[i].predict(X).cpu().numpy() for i in range(3)]
        pred = np.asarray(output)

        # 如果模型 1 和 2 的预测一致，则以它们的预测结果为准
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]

        # 返回最终的预测结果
        y_pred = pred[0]
        return y_pred
