import numpy as np
import sklearn
import pdb


class Tri_Training:
    def __init__(self, base_estimator_1, base_estimator_2, base_estimator_3):
        self.estimators = [base_estimator_1, base_estimator_2, base_estimator_3]

    def measure_error(self, X, y, j, k):
        # 输出模型 j 和 k 的编号
        print(f"Measuring error between model {j} and model {k}")

        # 获取模型 j 和 k 的预测结果
        j_pred = self.estimators[j].predict(X)
        k_pred = self.estimators[k].predict(X)

        # 输出部分预测结果，方便查看
        print(f"Model {j} predictions (first 10): {j_pred[:10]}")
        print(f"Model {k} predictions (first 10): {k_pred[:10]}")

        # 计算错误索引：模型 j 错误预测但 k 和 j 预测一致的样本
        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)

        # 输出错误样本数及其占比
        print(
            f"Number of samples where model {j} is wrong but {k} agrees: {sum(wrong_index)}"
        )

        # 计算模型 j 和 k 预测一致的样本总数
        same_pred_count = sum(j_pred == k_pred)
        print(
            f"Number of samples where both models predict the same: {same_pred_count}"
        )

        # 如果没有相同预测，避免除以零
        if same_pred_count == 0:
            print("No samples where both models predict the same, returning 0")
            return 0

        # 返回模型 j 出错但 k 同意预测的比例
        error_rate = sum(wrong_index) / same_pred_count
        print(f"Error rate for model {j} when both models agree: {error_rate}")

        return error_rate

    def fit(self, X, y, unlabeled_X):
        for i in range(3):
            sub_X, sub_y = sklearn.utils.resample(X, y)
            print("fitting estimator:", i)
            self.estimators[i].fit(sub_X, sub_y)
        e_prime = [0.5] * 3
        l_prime = [0] * 3
        e = [0] * 3
        update = [False] * 3
        lb_X, lb_y = [[]] * 3, [[]] * 3
        improve = True
        iter = 0
        while improve:
            iter += 1
            print("iteration:", iter)
            # pdb.set_trace()
            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False
                e[i] = self.measure_error(X, y, j, k)
                if e[i] < e_prime[i]:
                    print("predicting:", j)
                    ulb_y_j = self.estimators[j].predict(unlabeled_X)
                    print("predicting:", k)
                    ulb_y_k = self.estimators[k].predict(unlabeled_X)
                    lb_X[i] = unlabeled_X[ulb_y_j == ulb_y_k]
                    lb_y[i] = ulb_y_j[ulb_y_j == ulb_y_k]
                    if l_prime[i] == 0:
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    if l_prime[i] < len(lb_y[i]):
                        if e[i] * len(lb_y[i]) < e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                            lb_index = np.random.choice(
                                len(lb_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1)
                            )
                            lb_X[i], lb_y[i] = lb_X[i][lb_index], lb_y[i][lb_index]
                            update[i] = True
            for i in range(3):
                if update[i]:
                    print("updating estimator:", i)
                    self.estimators[i].fit(
                        np.append(X, lb_X[i], axis=0), np.append(y, lb_y[i], axis=0)
                    )
                    e_prime[i] = e[i]
                    l_prime[i] = len(lb_y[i])
            if update == [False] * 3:
                improve = False
        return

    def predict(self, X):
        print(f"predicting {len(X)} samples")
        output = [self.estimators[i].predict(X).cpu().numpy() for i in range(3)]
        pred = np.asarray(output)
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        y_pred = pred[0]
        return y_pred
