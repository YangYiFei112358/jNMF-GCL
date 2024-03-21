#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/6 10:57
# @Author  : YOUR_NAME
# @FileName: work1.py
# @Software: PyCharm

import math
import scipy.sparse as sp
from sklearn.cluster import KMeans
import numpy as np
from method import metric_all
from method.optimize import init_bl_fl, compute_neighbor, update_b_list, update_f_list, update_f_mat, update_z_mat, \
    compute_loss, adam


def read_files_func(file_paths):
    w_list = []
    degree_list = []
    laplacian_list = []

    for path in file_paths:
        wl_mat = np.load(path)
        w_list.append(wl_mat)

        dl_mat = np.diag(np.sum(wl_mat, axis=0))
        degree_list.append(dl_mat)

        laplacian_mat = dl_mat - wl_mat
        laplacian_list.append(laplacian_mat)

    return w_list, degree_list, laplacian_list


def read_labels_func(file_paths):
    labels = np.load(file_paths)
    return labels


def init_func(file_paths, para_list):
    reduce_dim = para_list['reduce_dim']
    layer_num = para_list['layer_num']
    theta = para_list['theta']

    w_list, degree_list, laplacian_list = read_files_func(file_paths)
    b_list = []
    f_list = []
    z_mat = np.zeros_like(w_list[0])
    for wl_mat in w_list:
        b_tmp, f_tmp = init_bl_fl(wl_mat, reduce_dim)
        b_list.append(b_tmp)
        f_list.append(f_tmp)
        z_mat += wl_mat

    f_mat = np.zeros_like(f_list[0])
    for fl_mat in f_list:
        f_mat += fl_mat

    f_mat = f_mat / layer_num
    z_mat = z_mat / layer_num

    nbr_dict = compute_neighbor(z_mat, para_list)

    return w_list, degree_list, laplacian_list, b_list, f_list, f_mat, z_mat, nbr_dict


def algorithm_func(file_paths, labels_path, para_list, save_file_path):
    # initialization of all variables
    w_list, degree_list, laplacian_list, b_list, f_list, f_mat, z_mat, nbr_dict = init_func(file_paths, para_list)
    labels = read_labels_func(labels_path)

    epochs = para_list['epochs']
    order = para_list['order']
    error_radio = math.pow(10, -order)

    cf = None
    loss_last = 1e16

    loss_list = []
    for epoch in range(epochs):
        b_list = update_b_list(w_list, b_list, f_list, para_list)
        f_list = update_f_list(w_list, b_list, f_list, f_mat, degree_list, para_list)
        f_mat = update_f_mat(f_list, f_mat, z_mat, para_list)
        grad_z_mat = update_z_mat(f_mat, z_mat, nbr_dict, para_list)
        z_mat, cf = adam(z_mat, grad_z_mat, cf)
        nbr_dict = compute_neighbor(z_mat, para_list)
        loss = compute_loss(w_list, b_list, f_list, f_mat, z_mat, laplacian_list, nbr_dict, para_list)

        output_sentence_1 = "[epcho][{:>3}]: loss is {:.4f}".format(epoch, loss)
        print(output_sentence_1)
        loss_list.append(loss)

        if abs((loss - loss_last) / loss) <= error_radio:
            print("The convergence condition meet!!!")
            break
        else:
            loss_last = loss

    cluster_number = len(np.unique(labels))
    c_mat = 0.5 * (np.fabs(z_mat) + np.fabs(z_mat.T))
    u, s, v = sp.linalg.svds(c_mat, k=cluster_number, which='LM')
    # Clustering
    kmeans = KMeans(n_clusters=cluster_number, random_state=7).fit(u)
    predict_labels = kmeans.predict(u)

    re = metric_all.ClusteringMetrics(predict_labels, labels)
    acc, nmi, ari, f1 = re.evaluation_cluster_model_from_label()
    output_sentence_2 = "Result: ACC={},NMI={},ARI={},F1={}.".format(acc, nmi, ari, f1)
    print(output_sentence_2)

    loss_path = save_file_path['loss_path']
    f_mat_path = save_file_path['f_mat_path']
    z_mat_path = save_file_path['z_mat_path']
    np.save(loss_path, loss_list)
    np.save(f_mat_path, f_mat)
    np.save(z_mat_path, z_mat)
    print("Done.")

    return predict_labels, acc, nmi, ari, f1


if __name__ == '__main__':
    # 需要调整的参数如下： miu, layer_nums
    mu = 80
    layer_nums = 5

    prefix = "E:/Dataset/homo_npy/" + str(mu) + "/layer"
    suffix = "/W.npy"
    data_file_paths = []
    for i in range(1, layer_nums + 1):
        data_file_paths.append(prefix + str(i) + suffix)

    parameter_dict = {'alpha': 1, 'beta': 0.1, 'gamma': 0, 'theta': 0.1,
                      'node_num': 1000, 'reduce_dim': 100, 'layer_num': layer_nums,
                      'epochs': 100, 'order': 3}

    gnd_path = "./data/LFR_labels.npy"

    output_prefix = "./output/"

    output_file_path = {
        'loss_path': output_prefix + str(mu) + '/loss.npy',
        'f_mat_path': output_prefix + str(mu) + '/f_mat.npy',
        'z_mat_path': output_prefix + str(mu) + '/z_mat.npy',
    }

    parameter_list = [0.1, 1, 10]
    for a in parameter_list:
        for b in parameter_list:
            for c in parameter_list:
                parameter_dict['alpha'], parameter_dict['beta'], parameter_dict['gamma'] = a, b, c
                output_sentence_4 = "alpha = {}, beta = {}, gamma = {}".format(a, b, c)
                print(output_sentence_4)
                algorithm_func(data_file_paths, gnd_path, parameter_dict, output_file_path)
    print("All Done.")
