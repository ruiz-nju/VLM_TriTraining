import numpy as np
import sklearn
from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import ClassifierMixin
from torch.utils.data.dataset import Dataset
import pdb


class Tri_Training:
    def __init__(self, base_estimator_1, base_estimator_2, base_estimator_3):
        self.estimators = [base_estimator_1, base_estimator_2, base_estimator_3]

    def measure_error(self, X, y, j, k):
        print("measure error:", j)
        j_pred = self.estimators[j].predict(X)
        print("measure error:", k)
        k_pred = self.estimators[k].predict(X)
        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)
        return sum(wrong_index) / sum(j_pred == k_pred)

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
