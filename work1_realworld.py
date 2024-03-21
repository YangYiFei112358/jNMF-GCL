#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 16:26
# @Author  : YOUR-NAME
# @FileName: work1_realworld.py
# @Software: PyCharm

import math
import os
import logger as l
import scipy.sparse as sp
from sklearn.cluster import KMeans
import numpy as np
from method import metric_all
from method.multi_layer_metric import compute_multi_layer_ass, compute_multi_layer_density
from method.optimize import init_bl_fl, compute_neighbor, update_b_list, update_f_list, update_f_mat, update_z_mat, \
    compute_loss, adam
import yaml
from path_file_real_world import path_real_world


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
        l.logger.info(output_sentence_1)
        loss_list.append(loss)

        if abs((loss - loss_last) / loss) <= error_radio:
            l.logger.info("The convergence condition meet!!!")
            break
        else:
            loss_last = loss

    cluster_number = para_list['clusters_num']
    c_mat = 0.5 * (np.fabs(z_mat) + np.fabs(z_mat.T))
    u, s, v = sp.linalg.svds(c_mat, k=cluster_number, which='LM')
    # Clustering
    kmeans = KMeans(n_clusters=cluster_number, random_state=7).fit(u)
    predict_labels = kmeans.predict(u)

    loss_path = save_file_path['loss_path']
    f_mat_path = save_file_path['f_mat_path']
    z_mat_path = save_file_path['z_mat_path']
    np.save(loss_path, loss_list)
    np.save(f_mat_path, f_mat)
    np.save(z_mat_path, z_mat)

    ass_array1, ass1 = compute_multi_layer_ass(w_list, predict_labels)
    l.logger.info(ass_array1)
    l.logger.info(ass1)
    temp_array1 = compute_multi_layer_density(w_list, predict_labels)
    l.logger.info(temp_array1)


if __name__ == '__main__':
    # 打印日志需要添加的
    output_data_path = "./"
    log_path = os.path.join(output_data_path, 'log')
    # 初始化日志
    l.initlog(log_path, )
    # l.logger.info()

    dataset_name_list = ['amazon', 'cancer', 'cellphone', 'dblp', 'p2p', 'cell_phone']

    # 指定 dataset
    dataset = 'cellphone'
    l.logger.info(dataset)

    directory = "./output_real_world/" + dataset + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 加载参数
    f = open('./yaml/real_world_info.yaml')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    parameters = cfg["real_world_datasets"]

    data_file_paths, parameter_dict, output_file_path = path_real_world(dataset,
                                                                        parameters[dataset]['node_num'],
                                                                        parameters[dataset]['reduce_dim'],
                                                                        parameters[dataset]['layer_num'],
                                                                        parameters[dataset]['clusters_num'])
    # para_list = [0.01, 0.1, 1, 10, 100]
    # for a in para_list:
    #     for b in para_list:
    #         for c in para_list:
    #             parameter_dict['alpha'], parameter_dict['beta'], parameter_dict['gamma'] = a, b, c
    #             output_sentence_4 = "alpha = {}, beta = {}, gamma = {}".format(a, b, c)
    #             print(output_sentence_4)
    #             algorithm_func_no_labels(data_file_paths, parameter_dict, output_file_path)
    parameter_list = [0.1, 1, 10]
    for a in parameter_list:
        for b in parameter_list:
            for c in parameter_list:
                parameter_dict['alpha'], parameter_dict['beta'], parameter_dict['gamma'] = a, b, c
                output_sentence_4 = "alpha = {}, beta = {}, gamma = {}".format(a, b, c)
                l.logger.info(output_sentence_4)
                algorithm_func(data_file_paths, "", parameter_dict, output_file_path)