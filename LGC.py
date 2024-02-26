import numpy as np
from collections import Counter
import math
from scipy.stats import poisson

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SuperNode:  ## --- for labeled node in graph
    id = 0
    default_lambda = 4

    def __init__(self, label, instances, bagging_lambda=None):
        self.label = label
        self.instances = instances
        _lambda = SuperNode.default_lambda if bagging_lambda is None else bagging_lambda
        self.bagging_lambdas = _lambda * np.ones(len(instances))
        self.size = len(instances)
        self.id = SuperNode.id
        SuperNode.id += 1
        return

    def update(self, x, bagging_lambda=None):
        self.instances = np.concatenate([self.instances, [x]])
        _lambda = SuperNode.default_lambda if bagging_lambda is None else bagging_lambda
        try:
            self.bagging_lambdas = np.concatenate([self.bagging_lambdas, [_lambda]])
        except:
            logger.error(
                "Error-update bagging_lambda: %s, bagging_lambdas: %s" % (bagging_lambda, self.bagging_lambdas))
            exit(-1)
        # self.instances.append(x)
        # self.centroid = (self.centroid * self.size + x) / (self.size + 1)
        self.size += 1
        return


class SemiLGCGraph:
    def __init__(self, graph_params, star_mesh=False):
        ## --- hyper-parameters

        self.buffer_size = 256
        self.iter_num = 100
        self.is_star_mesh = star_mesh
        self.is_bagging = True

        ## --- variables
        self.unlabeled_instances = []
        self.super_node_dict = {}  ## key = label, value = SuperNode
        self.super_node_list = []
        self.W = np.zeros(shape=(0, 0))
        self.D = np.zeros(shape=(0, 0))
        self.S = np.zeros(shape=(0, 0))  ## S = D^(-1/2) * W * D^(-1/2)

        self.Y_matrix = np.zeros(shape=(0, 0))
        self.F_matrix = np.zeros(shape=(0, 0))

        self.labeled_instances = []
        self.labeled_instance_weights = []
        self.label_data_y_collector = []  ## --- labeled data for model update
        self.label_data_collector_size = 128

        self.graph_params = graph_params
        self.alpha = self.graph_params['alpha']
        self.gamma = self.graph_params['gamma']  ## --- gamma: parameter of RBF kernel
        self.min_lambda = self.graph_params['min_lambda']  ## --- min_lambda: minimum value of lambda
        self.max_lambda = self.graph_params['max_lambda']

        self._sample_count = 0

        self.ratio = None

        print("alpha: %s, gamma: %s, min_lambda: %s, max_lambda: %s" % (
        self.alpha, self.gamma, self.min_lambda, self.max_lambda))
        return

    def get_true_W(self, X):
        W = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                W[i, j] = self._get_distance(X[i], X[j])
                if i == j:
                    W[i, j] = 0
        return W

    def _get_distance(self, x1, x2):
        K = -(np.linalg.norm(x1 - x2) ** 2) * self.gamma
        return np.exp(K)

    def _grow_graph_exist_supernode(self, x, y, weight=1):
        if not self.is_bagging:
            weight = 1
        class_num = len(self.super_node_dict)
        for j in range(len(self.unlabeled_instances)):
            _d = self._get_distance(x, self.unlabeled_instances[j])
            self.W[0, class_num + j] += _d * weight
            self.W[class_num + j, 0] += _d * weight
        ## Dii = Dii + new_Wi
        for j in range(len(self.D)):
            self.D[j, j] += self.W[j, 0]
        self.D[0, 0] = self.W[0].sum()
        ## --- cannot incremental update
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(self.D)))
        self.S = D_inv_sqrt.dot(self.W).dot(D_inv_sqrt)
        return

    def _grow_graph_new_unlabeled_node(self, X):
        m1, m2 = self.W.shape
        assert m1 == m2 and m1 > 0
        self.W = np.hstack((self.W, np.zeros((m1, 1))))
        self.W = np.vstack((self.W, np.zeros((1, m2 + 1))))
        class_num = len(self.super_node_dict)
        for j in range(len(self.unlabeled_instances)):
            _d = self._get_distance(X, self.unlabeled_instances[j])
            self.W[-1, class_num + j] = _d
            self.W[class_num + j, -1] = _d
        for k in range(len(self.super_node_list)):
            node = self.super_node_list[k]
            super_node = self.super_node_dict[node]
            _d = 0
            for l in range(len(super_node.instances)):
                _d += self._get_distance(X, super_node.instances[l]) * super_node.bagging_lambdas[l]
            self.W[-1, k] = _d
            self.W[k, -1] = _d
        self.W[-1, -1] = 0

        if len(self.unlabeled_instances) >= self.buffer_size:
            if self.is_star_mesh:
                self._remove_oldest_unlabeled_node_star_mesh()
            else:
                self._remove_oldest_unlabeled_node()

        ## D -- add new row and new column
        # self.D = np.vstack((np.hstack((self.D, np.zeros((len(self.D), 1)))), np.hstack((np.zeros((1, len(self.D))), np.zeros((1, 1))))))
        self.D = np.vstack((self.D, np.zeros((1, len(self.D)))))
        self.D = np.hstack((self.D, np.zeros((len(self.D), 1))))
        ## Dii = Dii + new_Wi
        for j in range(len(self.D)):
            self.D[j, j] += self.W[j, -1]
        self.D[-1, -1] = self.W[-1].sum()
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(self.D)))
        self.S = D_inv_sqrt.dot(self.W).dot(D_inv_sqrt)
        return

    def _remove_oldest_unlabeled_node_star_mesh(self):
        self.unlabeled_instances.pop(0)

        # update weight matrix
        self.class_num = len(self.super_node_dict)
        total_weight = np.sum(self.W[self.class_num, :])

        assert total_weight > 0, "total_weight = {} ".format(total_weight)
        # assert(len(self.sample_buffer) == self.buffer_size)
        self.unlabel_num = len(self.unlabeled_instances)
        for index1 in range(self.class_num + 1, self.class_num + self.unlabel_num):
            for index2 in range(self.class_num + 1, self.class_num + self.unlabel_num):
                if index1 != index2:
                    self.W[index1, index2] += self.W[self.class_num, index1] * self.W[self.class_num, index2] / \
                                              total_weight
                    self.D[index1, index1] += self.W[self.class_num, index1] * self.W[self.class_num, index2] / \
                                              (total_weight * 2)
                    self.D[index2, index2] += self.W[self.class_num, index1] * self.W[self.class_num, index2] / \
                                              (total_weight * 2)
        self.W = np.delete(self.W, self.class_num, axis=0)
        self.W = np.delete(self.W, self.class_num, axis=1)
        self.D = np.delete(self.D, self.class_num, axis=0)
        self.D = np.delete(self.D, self.class_num, axis=1)
        # self.labels = np.delete(self.labels, self.class_num, axis=0)
        return

    def _remove_oldest_unlabeled_node(self):
        self.unlabeled_instances.pop(0)
        self.class_num = len(self.super_node_dict)
        self.W = np.delete(self.W, self.class_num, axis=0)
        self.W = np.delete(self.W, self.class_num, axis=1)
        self.D = np.delete(self.D, self.class_num, axis=0)
        self.D = np.delete(self.D, self.class_num, axis=1)
        return

    def _grow_graph_new_labeled_node(self, X, y, weight=1):
        if not self.is_bagging:
            weight = 1
        m1, m2 = self.W.shape
        assert m1 == m2 and m1 > 0
        self.W = np.hstack((np.zeros((m1, 1)), self.W))
        self.W = np.vstack((np.zeros((1, m2 + 1)), self.W))
        # new_W = np.exp(-gamma * np.square(X - self.unlabeled_instances).sum(axis=1))
        class_num = len(self.super_node_dict)
        for j in range(len(self.unlabeled_instances)):
            _d = self._get_distance(X, self.unlabeled_instances[j])
            self.W[0, class_num + j] = _d * weight
            self.W[class_num + j, 0] = _d * weight
        self.W[-1, -1] = 0
        m1, m2 = self.D.shape
        assert m1 == m2 and m1 > 0
        self.D = np.hstack((np.zeros((m1, 1)), self.D))
        self.D = np.vstack((np.zeros((1, m2 + 1)), self.D))
        ## Dii = Dii + new_Wi
        for j in range(len(self.D)):
            self.D[j, j] += self.W[j, 0]
        self.D[0, 0] = self.W[0].sum()
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(self.D)))
        self.S = D_inv_sqrt.dot(self.W).dot(D_inv_sqrt)
        return

    def _optimize_F(self):
        n = len(self.super_node_list) + len(self.unlabeled_instances)
        Y_matrix = np.zeros((n, len(self.super_node_list)))
        for i in range(len(self.super_node_list)):
            Y_matrix[i, i] = 1
        F_matrix = Y_matrix.copy()
        ## F_matrix = alpha * S * F_matrix + (1 - alpha) * Y_matrix
        for i in range(self.iter_num):
            F_matrix = self.alpha * self.S.dot(F_matrix) + (1 - self.alpha) * Y_matrix
        self.F_matrix = F_matrix
        return

    def grow_graph_labeled(self, X, y, weight=1):
        if not self.is_bagging:
            weight = 1
        self.label_data_y_collector.append(y)
        if self.W.shape[0] == 0:
            self.W = np.array([[0]])
            self.D = np.array([[0]])
            self.super_node_dict[y] = SuperNode(y, [X], weight)
            self.super_node_list.insert(0, y)
            return
        if y in self.super_node_dict:
            # print("case2: grow graph with labeled node, in the supernode")
            self.super_node_dict[y].update(X, weight)
            self._grow_graph_exist_supernode(X, y, weight)
        else:
            # print("case3: grow graph with labeled node, new supernode")
            self.super_node_dict[y] = SuperNode(y, [X], weight)
            self.super_node_list.insert(0, y)

            self._grow_graph_new_labeled_node(X, y, weight)
        self._optimize_F()
        return

    def grow_graph_labeled_online_bagging(self, X, y):
        self.label_data_y_collector.append(y)
        _lambda = self._compute_lambda(y)
        sample_weight = poisson.rvs(_lambda, size=1).reshape(-1)[0]
        self.grow_graph_labeled(X, y, sample_weight)
        return

    def grow_graph_unlabeled(self, X):
        if self.W.shape[0] == 0:
            self.W = np.array([[0]])
            self.D = np.array([[0]])
            self.unlabeled_instances.append(X)
            return
        # print("case1: grow graph with unlabeled node")
        self.unlabeled_instances.append(X)
        self._grow_graph_new_unlabeled_node(X)
        self._optimize_F()
        return

    def _compute_lambda(self, y):
        # self._manage_recent_y_for_lambda(y)  ## --- function from lyf_model.py,but not used here
        most_common_y = Counter(self.label_data_y_collector).most_common(1)[0][0]
        most_common_y_count = Counter(self.label_data_y_collector).most_common(1)[0][1]
        this_y_count = Counter(self.label_data_y_collector)[y]
        # print(most_common_y, "most_common_y_count: %s, y: %s this_y_count: %s"%(most_common_y_count,y,this_y_count))
        try:
            _lambda = self.min_lambda + math.log10(most_common_y_count / this_y_count) * self.min_lambda
        except:
            logger.error("Error-compute _lambda error: most_common_y_count: %s, y: %s this_y_count: %s" % (
            most_common_y_count, y, this_y_count))
            exit(-1)
        return _lambda

    def predict(self, X_i, y_i):
        self.grow_graph_unlabeled(X_i)
        preds = self.F_matrix[-1:]
        pred_y = self.super_node_list[np.argmax(preds)]
        return pred_y, preds

    def predict_last_unlabel(self):
        # self.grow_graph(X,y)
        preds = self.F_matrix[-1:]
        pred_y = self.super_node_list[np.argmax(preds)]
        return pred_y, preds

    def _train_batch(self, X, y, label_state):
        class_list = np.unique(y)
        ## 1. get weight matrix W 
        W = self.get_true_W(X)
        ## 2. get D and S
        D = np.diag(np.sum(W, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        S = D_inv_sqrt.dot(W).dot(D_inv_sqrt)
        ## 3. get Y_matrix, n*c, yij = 1 if i is labeled and classified as j
        n = len(y)
        c = len(class_list)
        Y_matrix = np.zeros((n, c))
        for i in range(n):
            if label_state[i] == 1:
                for j in range(c):
                    if y[i] == class_list[j]:
                        Y_matrix[i, j] = 1
        ## 4. get F_matrix
        F_matrix = Y_matrix.copy()
        for i in range(self.iter_num):
            F_matrix = self.alpha * S.dot(F_matrix) + (1 - self.alpha) * Y_matrix

        ## 5. get pred_y
        unlabeled_y = []
        pred_y = []
        for i in range(n):
            if label_state[i] == 0:
                unlabeled_y.append(y[i])
                pred_y.append(class_list[np.argmax(F_matrix[i])])
        unlabeled_y = np.array(unlabeled_y)
        pred_y = np.array(pred_y)

        ## get accuracy
        accuracy = sum(pred_y == unlabeled_y) / n
        print("accuracy: ", accuracy)
        return pred_y, unlabeled_y
