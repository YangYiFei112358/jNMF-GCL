#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2024/3/4 19:57
# @Author  : YOUR-NAME
# @FileName: test_logging.py
# @Software: PyCharm

import os
import logger as l
import numpy as np

def compute_neighbor(z_mat):
    theta = 0.5
    nbr_dict = {}
    row_num, col_num = z_mat.shape
    # hang_maxlist = np.max(z_mat, axis=0)
    # thod = hang_maxlist * theta

    # for i in range(row_num):
    #
    #     ind = np.argpartition(z_mat[i], int(-1 * theta * row_num))
    #     nbr_dict[i] = set(ind)
    #     # nbr_dict[i] = set()
    #     # for j in range(col_num):
    #     #     if z_mat[i][j] >= thod[i] and j != i:
    #     #         nbr_dict[i].add(j)
    # return nbr_dict
    # ----------这是一条分界线----------
    for i in range(row_num):
        k = int(-1 * theta * row_num)
        ind = np.argpartition(z_mat[i], k)
        # ind = ind[k:]
        nbr_dict[i] = set(ind)
    return nbr_dict

if __name__ == '__main__':
    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(a[1])
    # b = compute_neighbor(a)
    # print(b)


    # import numpy as np
    #
    # # 创建一个数组
    # arr = np.array([3, 1, 4, 2, 5])
    #
    # # 找出前3大数字的索引
    # k = 3
    # indices = np.argpartition(arr, -k)[-k:]
    #
    # print("Indices of top", k, "elements:", indices)
    # print("Top", k, "elements:", arr[indices])


    output_data_path = "D:/Code/code_reformat/"
    log_path = os.path.join(output_data_path, 'log')

    # 初始化日志
    l.initlog(log_path, )
    for i in range(100):
        l.logger.info(f'run with args = {i}')

    print('PyCharm')

