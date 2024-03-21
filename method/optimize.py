#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/5 17:05
# @Author  : YOUR-NAME
# @FileName: optimize.py
# @Software: PyCharm

import numpy as np


def init_bl_fl(adj, k):
    # this method aims to get init of B, F
    [u_mat, s_mat, v_mat] = np.linalg.svd(adj)
    # u_mat: m*m
    # s_mat: min(m,n)*1
    # v_mat: n*n
    u_mat = u_mat[:, :k]
    s_mat = np.diag(s_mat[:k])
    v_mat = v_mat.T
    v_mat = v_mat[:, :k]
    v_mat = v_mat.T
    b_mat = abs(u_mat.dot(np.sqrt(s_mat)))
    f_mat = abs(np.sqrt(s_mat).dot(v_mat))
    return b_mat, f_mat


def update_b_list(w_list, b_list, f_list, para_list, my_eps=1e-10):
    # b_list contains basis matrix of all layers, we use one of them by index
    # b_list : layer_num * 1
    # b_mat, b_list[k]: node_num * reduce_dim
    node_num = para_list['node_num']
    layer_num = para_list['layer_num']

    row_num = node_num
    for k in range(layer_num):
        numerators = np.dot(w_list[k], f_list[k].T)
        denominators = np.dot(np.dot(b_list[k], f_list[k]), f_list[k].T)
        b_list[k] = b_list[k] * numerators / np.maximum(denominators, my_eps)
        # normalization 列归一化
        # for i in range(col_num):
        #     norm_L2 = np.linalg.norm(B[:, i])
        #     if norm_L2 > 0:
        #         B[:, i] = B[:, i] / norm_L2

        # normalization 行归一化
        for i in range(row_num):
            norm_l2 = np.linalg.norm(b_list[k][i, :])
            if norm_l2 > 0:
                b_list[k][i, :] = b_list[k][i, :] / norm_l2
    return b_list


def update_f_list(w_list, b_list, f_list, f_mat, degree_list, para_list, my_eps=1e-10):
    # f_list contains represent matrix of all layers
    # we use one of them by index
    # f_list : layer_num * 1
    # f_list[k]: reduce_dim * node_num

    node_num = para_list['node_num']
    alpha = para_list['alpha']
    layer_num = para_list['layer_num']

    col_num = node_num
    for k in range(layer_num):
        # 计算分子
        numerators = np.dot(b_list[k].T, w_list[k])
        numerators += np.dot(f_list[k], w_list[k])
        numerators += alpha * f_mat
        # 计算分母
        denominators = np.dot(np.dot(b_list[k].T, b_list[k]), f_list[k])
        denominators += np.dot(f_list[k], degree_list[k])
        denominators += alpha * f_list[k]
        # update F[k]
        f_list[k] = f_list[k] * numerators / np.maximum(denominators, my_eps)

        # normalization 行归一化
        # for i in range(row_num):
        #     norm_L2 = np.linalg.norm(F[k][i, :])
        #     if norm_L2 > 0:
        #         F[k][i, :] = F[k][i, :] / norm_L2

        # normalization 列归一化
        for i in range(col_num):
            norm_l2 = np.linalg.norm(f_list[k][:, i])
            if norm_l2 > 0:
                f_list[k][:, i] = f_list[k][:, i] / norm_l2
    return f_list


def update_f_mat(f_list, f_mat, z_mat, para_list, my_eps=1e-10):
    # f_mat is shared feature of vertices in multi-layer networks
    # f_mat: reduce_dim * node_num
    reduce_dim = para_list['reduce_dim']
    node_num = para_list['node_num']
    layer_num = para_list['layer_num']
    alpha = para_list['alpha']
    beta = para_list['beta']

    row_num = reduce_dim
    col_num = node_num

    # 计算分子
    numerators = np.zeros((row_num, col_num))
    for k in range(layer_num):
        numerators += alpha * f_list[k]
    numerators += beta * np.dot(f_mat, z_mat + z_mat.T)
    # 计算分母
    denominators = (alpha * layer_num + beta) * np.identity(node_num)
    denominators = denominators + beta * np.dot(z_mat, z_mat.T)
    denominators = np.dot(f_mat, denominators)
    f_mat = f_mat * numerators / np.maximum(denominators, my_eps)

    # normalization 行归一化
    # for i in range(row_num):
    #     norm_l2 = np.linalg.norm(f_mat[i, :])
    #     if norm_l2 > 0:
    #         f_mat[i, :] = f_mat[i, :] / norm_l2

    # normalization 列归一化
    for i in range(col_num):
        norm_l2 = np.linalg.norm(f_mat[:, i])
        if norm_l2 > 0:
            f_mat[:, i] = f_mat[:, i] / norm_l2
    return f_mat


def adam(w, dw, config=None):
    """
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    m = config['m']
    v = config['v']
    t = config['t'] + 1
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']
    learning_rate = config['learning_rate']

    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    mb = m / (1 - beta1 ** t)
    vb = v / (1 - beta2 ** t)
    next_w = w - learning_rate * mb / (np.sqrt(vb) + epsilon)

    config['m'] = m
    config['v'] = v
    config['t'] = t
    # print(config['t'])

    return next_w, config


def update_z_mat(f_mat, z_mat, nbrs_dic, para_list):
    node_num = para_list['node_num']
    beta = para_list['beta']
    gamma = para_list['gamma']

    row_num = node_num
    col_num = node_num

    grad_11 = -np.dot(f_mat.T, f_mat)
    grad_12 = np.dot(np.dot(f_mat.T, f_mat), z_mat)
    grad_1 = 2 * beta * (grad_11 + grad_12)

    grad_2 = np.zeros((node_num, node_num))
    for i in range(row_num):
        # z_mat[i] 是 z_mat 的一行
        k0 = np.exp(z_mat[i]).sum() - np.exp(z_mat[i][i])
        neighbor_num = len(nbrs_dic[i])
        for j in range(col_num):
            if j in nbrs_dic[i]:
                grad_2[i][j] = (-1 + neighbor_num * np.exp(z_mat[i][j]) / k0)
            else:
                grad_2[i][j] = (neighbor_num * np.exp(z_mat[i][j]) / k0)
    grad = gamma * grad_2 + grad_1
    return grad


def compute_neighbor(z_mat, para_list):
    theta = para_list['theta']
    nbr_dict = {}
    row_num, col_num = z_mat.shape
    for i in range(row_num):
        k = int(theta * row_num)
        # 按照从小到大的顺序进行排列
        ind = np.argpartition(z_mat[i], -k)
        # 从后往前取k个索引，表示前k大值的索引
        ind = ind[-k:]
        nbr_dict[i] = set(ind)
    return nbr_dict


def compute_loss(w_list, b_list, f_list, f_mat, z_mat, laplacian_list, nbrs_dic, para_list):
    layer_num = para_list['layer_num']
    node_num = para_list['node_num']

    alpha = para_list['alpha']
    beta = para_list['beta']
    gamma = para_list['gamma']

    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    for k in range(layer_num):
        temp = w_list[k] - np.dot(b_list[k], f_list[k])
        loss1 += np.linalg.norm(temp) ** 2
        loss2 += np.trace(np.dot(f_list[k], np.dot(laplacian_list[k], f_list[k].T)))
        loss3 += alpha * (np.linalg.norm(f_list[k] - f_mat) ** 2)

    loss4 = beta * np.linalg.norm(f_mat - np.dot(f_mat, z_mat)) ** 2

    loss5 = 0.0
    for i in range(node_num):
        k0 = np.exp(z_mat[i]).sum() - np.exp(z_mat[i][i])
        loss_nbr = 0
        for j in nbrs_dic[i]:
            loss_nbr = loss_nbr - np.log(np.exp(z_mat[i][j]) / k0)
        loss5 += gamma * loss_nbr
    loss = loss1 + loss2 + loss3 + loss4 + loss5
    return loss
