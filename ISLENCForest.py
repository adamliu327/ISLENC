# -*- coding: utf-8 -*-

import numpy as np
import random

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random_seed = 0


class Node:
    def __init__(self, node_id=None,
                 feature_data=None, center=None, radius=0,
                 selected_feature=None, split_point=None,
                 #  lson_id=None, lson_id=None,
                 #  parent_id=None,
                 parent_node=None,
                 #  lcenter=None, rcenter=None, lson=None, rson=None,
                 root_dist=0, is_leaf=False):
        # self.id           = node_id           # node id
        self.feature_data = feature_data  # feature vector
        self.center = center  # the center of current node
        self.radius = radius  # the range that current node covers
        self.selected_feature = selected_feature  # selected feature id to divide instances
        self.split_point = split_point  # split point to divide instances

        self.size = self.feature_data.shape[0]  # number of instances in the node
        self.parent_node = parent_node  # parent node

        self.lson_node = None  # left son node
        self.rson_node = None  # right son node

        self.root_dist = root_dist  # distance between current node and root node, namely the depth of current node
        self.is_leaf = is_leaf  # whether current node is leaf node


class Tree:
    random_seed = 0

    def __init__(self, feature_data=None, params=None):
        self.feature_data = feature_data  # all feature vector of current tree
        self.tree_size = feature_data.shape[0]  # number of instances in current tree
        self.max_tree_size = params['max_tree_size']  ## number of instances in current tree

        self.k_feature = 1  ## the same with iForest
        self.max_height = int(np.ceil(np.log2(self.max_tree_size)))  # maximum tree height, which is fixed in advance
        self.min_leaf_size = params['min_leaf_size']  ## minimum instances in leaf node, which is fixed in advance

        self.nodes = []  # store each nodes
        self.leaf_type = [0, 0, 0, 0]
        self.leaf_nodes = []  # store leaf nodes

        self.node_id_count = 0  # count the number of nodes

        self.is_half_leaf_max_height = False  # whether half of leaf nodes are max height

        self.avg_path_length = 0  # average path length of all instances
        self.init_tree()
        return

    def init_tree(self):
        self.root_node = self.build_tree(self.feature_data, 0)  # build tree

        self.leaf_dist = []
        self.node_dist = []
        for node in self.nodes:
            self.node_dist.append(node.root_dist)
        self.node_dist_mean = np.mean(self.node_dist)
        self.compute_average_path_length()
        return

    def is_duplicate_feature_data(self, fea):
        """ check if all instances in current node are the same """
        n, dim1 = fea.shape[0], fea.shape[1]
        for i in range(1, n):
            if sum(fea[i] == fea[0]) != dim1:
                return False
        return True

    def get_distance(self, x1, x2):
        assert x1.shape == x2.shape, "x1.shape = {} x2.shape = {}".format(x1.shape, x2.shape)
        return np.linalg.norm(x1 - x2) ** 2

    def create_leaf(self, feature, height, parent):
        leaf = Node(feature_data=feature, root_dist=height, parent_node=parent, is_leaf=True)
        x = feature

        leaf.center = np.mean(x, axis=0)
        leaf.radius = 0
        for i in range(x.shape[0]):
            d = self.get_distance(x[i], leaf.center)
            leaf.radius = max(leaf.radius, d)
        return leaf

    def random_select(self, n, m):
        assert (n >= m)
        np.random.seed(Tree.random_seed)
        Tree.random_seed += 1
        selected = np.random.choice(n, m, replace=False)
        selected = sorted(selected)
        return selected

    def build_tree(self, feature_data, height, parent=None):
        cur = len(self.nodes)
        n, dim1 = feature_data.shape[0], feature_data.shape[1]

        if height >= self.max_height or n <= self.min_leaf_size or self.is_duplicate_feature_data(
                feature_data):  ## --- create leaf
            if height >= self.max_height:
                self.leaf_type[0] += 1
            elif n <= self.min_leaf_size:
                self.leaf_type[1] += 1
            elif n == 0:
                self.leaf_type[2] += 1
            else:
                self.leaf_type[3] += 1

            if n == 0:
                return None
            # elif n == 1:
            #     assert(center_parent is not None and radius_parent is not None)
            #     _node = Node(feature_data=feature_data, center=center_parent, radius=radius_parent,
            #                            root_dist=height, is_leaf=True)
            #     self.nodes.append(_node)
            #     self.leaf_nodes.append(_node)
            else:
                _node = self.create_leaf(feature_data, height, parent)
                self.nodes.append(_node)
                self.leaf_nodes.append(_node)
            return _node

        inter_node = Node(feature_data=feature_data, root_dist=height)
        inter_node.center = np.mean(feature_data, axis=0)
        inter_node.radius = 0

        is_same_feature = True
        while is_same_feature:
            inter_node.selected_feature = self.random_select(dim1, self.k_feature)
            x = feature_data[:, inter_node.selected_feature]
            if min(x) != max(x):
                is_same_feature = False

        ## randomly select a split point in the range of [min(x), max(x)]
        random.seed(Tree.random_seed)
        Tree.random_seed += 1
        split_point = random.uniform(min(x), max(x))
        inter_node.split_point = split_point
        left_data, right_data = [], []
        for i in range(n):
            x_i_feature = feature_data[i, inter_node.selected_feature]
            if x_i_feature <= split_point:
                left_data.append(i)
            else:
                right_data.append(i)

            x_i = feature_data[i, :]
            dis = self.get_distance(x_i, inter_node.center)
            inter_node.radius = max(inter_node.radius, dis)

        self.nodes.append(inter_node)
        inter_node.lson_node = self.build_tree(feature_data[left_data, :], height + 1, inter_node)
        inter_node.rson_node = self.build_tree(feature_data[right_data, :], height + 1, inter_node)
        return inter_node

    def get_path_length(self, x_fea):
        node = self.find_leaf(x_fea)
        return node.root_dist

    def compute_average_path_length(self):
        sum_path_length = 0
        total_size = 0
        for node in self.leaf_nodes:
            sum_path_length += node.root_dist * node.size
            total_size += node.size
        self.avg_path_length = sum_path_length / total_size
        return self.avg_path_length

    def update_tree_online(self, x):  ## x is a single sample
        self.tree_size += 1
        node = self.find_leaf(x)
        node.feature_data = np.vstack((node.feature_data, x))
        node.size += 1
        node.center = np.mean(node.feature_data, axis=0)
        node.radius = 0
        for i in range(node.feature_data.shape[0]):
            d = self.get_distance(node.feature_data[i], node.center)
            node.radius = max(node.radius, d)
        height = node.root_dist

        if height >= self.max_height or node.size <= self.min_leaf_size or self.is_duplicate_feature_data(
                node.feature_data):  ## --- create leaf
            return
        else:
            is_same_feature = True
            while is_same_feature:
                node.selected_feature = self.random_select(node.feature_data.shape[1], self.k_feature)
                x = node.feature_data[:, node.selected_feature]
                if min(x) != max(x):
                    is_same_feature = False
            random.seed(Tree.random_seed)
            Tree.random_seed += 1
            split_point = random.uniform(min(x), max(x))
            node.split_point = split_point
            node.is_leaf = False
            left_data, right_data = [], []
            for i in range(node.size):
                x_i_feature = node.feature_data[i, node.selected_feature]
                if x_i_feature <= split_point:
                    left_data.append(i)
                else:
                    right_data.append(i)
            node.lson_node = self.build_tree(node.feature_data[left_data, :], height + 1, node)
            node.rson_node = self.build_tree(node.feature_data[right_data, :], height + 1, node)
            self.reset_leaf_nodes()
        return

    def reset_leaf_nodes(self):
        self.leaf_nodes = []
        _max_height_leaf_count = 0
        for node in self.nodes:
            if node.is_leaf:
                self.leaf_nodes.append(node)
                if node.root_dist >= self.max_height:
                    _max_height_leaf_count += 1
        if _max_height_leaf_count > len(self.leaf_nodes) // 2:
            self.is_half_leaf_max_height = True
        return

    def find_leaf(self, x):  ## x.shape = (1, dim)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        assert x.shape[0] == 1
        x = x[0]
        return self._find_leaf(self.root_node, x)

    def _find_leaf(self, node, x):
        if node.is_leaf:
            return node
        else:
            # print(node)
            if x[node.selected_feature] < node.split_point:
                if node.lson_node is None:
                    print(node)
                    raise Exception("node.lson_node is None")
                return self._find_leaf(node.lson_node, x)
            else:
                if node.rson_node is None:
                    print(node)
                    raise Exception("node.rson_node is None")
                return self._find_leaf(node.rson_node, x)


class ISLENCForest:
    random_seed = 0

    def __init__(self, feature_data=None, weight=None, params=None, verbose=False):
        self.verbose = verbose
        feature_data = np.array(feature_data)
        weight = [1 for i in range(feature_data.shape[0])] if weight is None else weight
        self.params = params

        assert feature_data.shape[0] == len(weight)

        self.feature_data = feature_data  # type: np.ndarray
        total_weight = sum(weight)
        self.weight = [i / total_weight for i in weight]  # type: list

        # self.max_tree_size = min(feature_data.shape[0], params['max_tree_size'])
        self.max_tree_size = params['max_tree_size']
        self.max_buffer_size = params['max_buffer_size']

        # self.k_fea = (feature_data.shape[1] + 1) // 2             
        self.max_height = int(np.ceil(np.log2(self.max_tree_size)))
        self.min_leaf_size = params['min_leaf_size']
        self.tree_num = params['tree_num']
        self.anomaly_score_threshold = params['anomaly_score_threshold']

        self.is_half_leaf_max_height = False  # whether half of leaf nodes are max height

        self._print_params()
        # self.build_forest()
        self.tree_size = 0

    def _print_params(self):
        if self.verbose: logger.info("tree_size: %s", self.max_tree_size)
        # logger.info("k_fea: %s", self.k_fea)
        if self.verbose: logger.info("max_height: %s", self.max_height)
        if self.verbose: logger.info("min_leaf_size: %s", self.min_leaf_size)
        if self.verbose: logger.info("number of trees: %s", self.tree_num)
        return

    def init_buffer(self):
        buffer_size = min(self.feature_data.shape[0], self.max_tree_size)
        indx = np.random.choice(self.feature_data.shape[0], buffer_size, replace=False)
        self.buffer_data = self.feature_data[indx, :]
        return

    def update(self, sample):
        self.update_buffer(sample)
        self.online_update(sample)
        # logger.info("buffer_size: %s", self.buffer_data.shape[0])
        if self.buffer_data.shape[0] >= self.max_buffer_size:  ## --- update here for concept drift
            self.trunk_update_by_buffer()
            logger.info("trunk_update_by_buffer")
        # else:
        # logger.info("self.buffer_data.shape[0]%s < self.max_buffer_size%s", self.buffer_data.shape[0], self.max_buffer_size)
        return

    def update_buffer(self, sample):
        self.buffer_data = np.vstack((self.buffer_data, sample))
        return

    def online_update(self, sample):
        ## case 1: no need to update
        if self.tree_size >= self.max_tree_size:  # no need to update
            # logger.info("tree_size >= max_tree_size, no need to update")
            return

        ## case 2: update tree by sample
        self.tree_size += 1
        _tree_max_height_leaf_count = 0
        for tree_id in range(self.tree_num):
            self.trees[tree_id].update_tree_online(sample)
            self.trees[tree_id].compute_average_path_length()
            if self.trees[tree_id].is_half_leaf_max_height:
                _tree_max_height_leaf_count += 1
        if _tree_max_height_leaf_count > self.tree_num // 2:
            self.is_half_leaf_max_height = True
        # logger.info("tree_size updated to %s", self.tree_size)
        return

    def trunk_update_by_buffer(self):
        self.feature_data = self.buffer_data.copy()
        self.build_forest()
        assert self.buffer_data.shape[0] >= self.max_tree_size
        indx = np.random.choice(self.buffer_data.shape[0], self.max_tree_size, replace=False)
        self.buffer_data = self.buffer_data[indx, :]
        return

    def trunk_update(self, samples):
        samples = np.array(samples)
        self.feature_data = samples.copy()  ##
        self.build_forest()
        return

    def trunk_update(self, samples, weight):
        samples = np.array(samples)
        self.feature_data = samples.copy()  ##
        total_weight = sum(weight)
        self.weight = [i / total_weight for i in weight]
        self.build_forest()
        return

    def select_samples(self):
        n = self.feature_data.shape[0]
        assert n >= self.max_tree_size
        np.random.seed(ISLENCForest.random_seed)
        ISLENCForest.random_seed += 1
        # selected = np.random.choice(n, self.max_tree_size, replace=False, p=self.weight)
        selected = np.random.choice(n, self.max_tree_size, replace=False)
        selected = sorted(selected)
        return selected

    def build_forest(self):
        self.trees = []
        for tree_id in range(self.tree_num):
            if self.feature_data.shape[0] <= self.max_tree_size:
                selected_id = [i for i in range(self.feature_data.shape[0])]
            else:
                selected_id = self.select_samples()
            self.tree_size = len(selected_id)
            selected_feature_data = self.feature_data[selected_id, :]
            cur_tree = Tree(selected_feature_data, params=self.params)
            self.trees.append(cur_tree)
        return

    def get_harmonic_number(self, n):
        return np.log(n) + 0.5772156649

    def predict_anomaly_iForest(self, x_fea):
        avg_tree_size = np.mean([tree.tree_size for tree in self.trees])
        avg_instance_path_length = np.mean([tree.get_path_length(x_fea) for tree in self.trees])
        c_n = 2 * self.get_harmonic_number(avg_tree_size - 1) - 2 * (avg_tree_size - 1) / avg_tree_size
        score = 2 ** (-avg_instance_path_length / c_n)
        return score > self.anomaly_score_threshold, score
