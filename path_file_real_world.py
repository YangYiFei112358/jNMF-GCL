#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 18:48
# @Author  : YOUR-NAME
# @FileName: path_file_real_world.py
# @Software: PyCharm

def path_real_world(dataset, node_num, reduce_dim, layer_num, clusters_num):
    dataset_name_list = ['amazon', 'cancer', 'cellphone', 'dblp', 'p2p', 'cell_phone']
    if dataset not in dataset_name_list:
        print("dataset name error!!!")
        return False

    my_para_dict = {'alpha': 1, 'beta': 1, 'gamma': 1, 'theta':0.1,
                    't': 0.5, 'epochs': 20, 'order': 3,
                    'node_num': node_num, 'reduce_dim': reduce_dim, 'layer_num': layer_num,
                    'clusters_num': clusters_num}
    prefix = "E:/Dataset/real_world/" + dataset + "/W"
    suffix = ".npy"
    layer_nums = my_para_dict['layer_num']
    my_data_file_paths = []
    for i in range(1, layer_nums + 1):
        abs_path = prefix + str(i) + suffix
        my_data_file_paths.append(abs_path)

    output_prefix = "./output_real_world/" + dataset + "/"
    my_output_file_path = {
        'pred_labels': output_prefix + '/pred_labels.npy',
        'loss_path': output_prefix + '/loss.npy',
        'f_mat_path': output_prefix + '/f_mat.npy',
        'z_mat_path': output_prefix + '/z_mat.npy',

        'ass_mat_path': output_prefix + '/ass_mat.npy',
        'ass_list_path': output_prefix + '/ass_list.npy',
        'density_mat_path': output_prefix + '/density_mat.npy',
    }
    return my_data_file_paths, my_para_dict, my_output_file_path
