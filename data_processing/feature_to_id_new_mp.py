#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2015, Inc. All Rights Reserved
#
################################################################################
"""
This module maps feature string to id string.(多进程版本)

Authors: Wang Shijun
Date:    2015/07/08 23:00:00

Outline:
@@ collect feature （多进程，每个进程读取一个数据文件，统计每个feature在该文件中的个数）
@@ merge feature （单进程，汇总每个文件中的feature个数）
@@ order feature （单进程，按出feature现次数排序，次数大的排在前面，其在列表中的位置即为其id的10进制表示）
@@ build map （多进程，每个进程将其分到的feature的id转换为93进制）
@@ merge map （单进程，汇总所有feature到id的映射关系）
@@ do mapping （多进程，每个进程读取一个数据文件，将其中出现的feature映射成id，并写入新的文件）
"""

import os
import datetime
import Queue
import multiprocessing as mp

def count_feature_partial(file_block):
    """
    collect feature （多进程，每个进程读取一个数据文件，统计每个feature在该文件中的个数）
    :param file_block:
    :return:
    """
    feature_dict = dict()
    for _file in [file_block]:
        print '{time} --> '.format(time=datetime.datetime.now()) + _file
        with open(_file) as f:
            for line in f:
                features = line.strip().split(',')
                for feature in features[2:]:
                    if feature in feature_dict:
                        feature_dict[feature] += 1
                    else:
                        feature_dict[feature] = 1
    return feature_dict

def merge_feature_count(fcs):
    """
    merge feature （单进程，汇总每个文件中的feature个数）
    :param fcs:
    :return:
    """
    f_count = dict()
    for fc in fcs:
        for f in fc:
            if f in f_count:
                f_count[f] += 1
            else:
                f_count[f] = 1
    return f_count

def get_id(count):
    """
    获取count对应的可用ID
    :return:
    """
    # ID可用的符号的开始
    sign_start = 33  # 符号'!'
    # ID不可用的符号
    eliminated_sign = [',']
    # ID的总数目
    sign_num = 94 - len(eliminated_sign)  # 94+33-1=126 符号'~'

    if count == 0:
        return str(unichr(sign_start))
    _id = ''
    while count:
        count, digit = divmod(count, sign_num)
        for sign in sorted(eliminated_sign):  # 跳过非法符号
            _tmp = ord(sign) - sign_start
            if digit >= _tmp:
                digit += 1
        _id = str(unichr(sign_start + digit)) + _id
    return _id

def build_map(f_n):
    """
    build map （多进程，每个进程将其分到的feature的id转换为93进制）
    :param f_n:
    :return:
    """
    return f_n[0], get_id(f_n[1])  # 返回tuple类型

def output_map(path, f_id_maps):
    """
    输出映射关系
    :return:
    """
    _dir = os.path.join(path, 'map')
    if os.path.isdir(_dir):
        pass
    else:
        os.mkdir(_dir)
    output_file = os.path.join(path, 'map', 'feature_id_map')

    f_id_maps_list = [it[1] + '\t' + it[0] + '\n' for it in f_id_maps]  # 先value后key
    with open(output_file, 'w') as f:
        f.writelines(f_id_maps_list)

def do_mapping(f_i_dict, f_queue):
    """
    do mapping （多进程，每个进程读取一个数据文件，将其中出现的feature映射成id，并写入新的文件）
    :param f_i_dict:
    :param f_queue:
    :return:
    """
    while 1:
        try:
            _file = f_queue.get_nowait()
            print '{time} --> '.format(time=datetime.datetime.now()) + _file

            path, filename = os.path.split(_file)
            _dir = os.path.join(path, 'zipped')
            if os.path.isdir(_dir):
                pass
            else:
                os.mkdir(_dir)
            output_file = os.path.join(path, 'zipped', filename)

            with open(_file) as file_in, open(output_file, 'w') as file_out:
                for line in file_in:
                    features = line.strip().split(',')
                    file_out.writelines(','.join(features[:2]) + ',')
                    _id = [f_i_dict[feature] for feature in features[2:]]
                    file_out.writelines(','.join(_id) + '\n')
            print '{time} --> '.format(time=datetime.datetime.now()) + _file + ' done'

        except Queue.Empty:
            break


if __name__ == '__main__':

    import argparse

    # 命令行参数设置
    parser = argparse.ArgumentParser(description='Maps feature string to id string.')
    parser.add_argument('file_path', metavar='File Path', help='data files')
    parser.add_argument("nop", help="number of processors", type=int)
    args = parser.parse_args()

    # 获取文件列表
    files = []
    for fs in os.listdir(args.file_path):
        full_file_name = os.path.join(args.file_path, fs)
        if os.path.isfile(full_file_name) and fs[0] != '.':
            files.append(full_file_name)

    # 开始计时
    start_time = datetime.datetime.now()

    # 每个文件分别构造feature_dict
    print '{time} --> collecting feature'.format(time=datetime.datetime.now())
    mm_start_time = datetime.datetime.now()
    pool = mp.Pool(args.nop)
    feature_counts = pool.map(count_feature_partial, files)  # 多进程
    pool.close()
    pool.join()
    mm_end_time = datetime.datetime.now()
    print '@@ collect feature time: ', mm_end_time - mm_start_time

    # 合并每个文件中的feature_dict
    print '{time} --> merging feature'.format(time=datetime.datetime.now())
    mm_start_time = datetime.datetime.now()
    feature_count = merge_feature_count(feature_counts)  # 此处必须用单进程，因此比较费时
    mm_end_time = datetime.datetime.now()
    print '@@ merge feature time: ', mm_end_time - mm_start_time

    # 按出现次数对feature排序
    print '{time} --> ordering feature'.format(time=datetime.datetime.now())
    mm_start_time = datetime.datetime.now()
    feature_list = [str(fc[0]) for fc in sorted(feature_count.items(), lambda x, y: cmp(y[1], x[1]))]  # 次数高的排在前面
    # feature_list = feature_count.keys()  # 自然顺序
    feature_pos_list = [(item, index) for index, item in enumerate(feature_list)]
    mm_end_time = datetime.datetime.now()
    print '@@ order feature time: ', mm_end_time - mm_start_time

    # 对feature按进程数切分，每个进程分别计算其负责部分的构造feature到id的映射
    print '{time} --> building map'.format(time=datetime.datetime.now())
    mm_start_time = datetime.datetime.now()
    pool = mp.Pool(args.nop)
    feature_id_maps = pool.map(build_map, feature_pos_list, len(feature_pos_list) / args.nop + 1)  # 多进程
    pool.close()
    pool.join()
    output_map(args.file_path, feature_id_maps)
    mm_end_time = datetime.datetime.now()
    print '@@ build map time: ', mm_end_time - mm_start_time

    # 合并map
    print '{time} --> merging map'.format(time=datetime.datetime.now())
    mm_start_time = datetime.datetime.now()
    # manager = mp.Manager()
    # feature_id_map = manager.dict(feature_id_maps)  # 从tuple list转换为dict
    feature_id_map = dict(feature_id_maps)  # 从tuple list转换为dict (没有对feature_map的写操作，不需要Manager.dict)
    mm_end_time = datetime.datetime.now()
    print '@@ merge map time: ', mm_end_time - mm_start_time

    # 将数据集文件中的feature替换为id
    print '{time} --> doing mapping'.format(time=datetime.datetime.now())
    mm_start_time = datetime.datetime.now()
    file_queue = mp.Queue()
    for f in files:
        file_queue.put(f)
    jobs = [mp.Process(target=do_mapping, args=(feature_id_map, file_queue)) for i in range(args.nop)]  # 多进程
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    mm_end_time = datetime.datetime.now()
    print '@@ do mapping time: ', mm_end_time - mm_start_time

    # 结束计时
    end_time = datetime.datetime.now()
    print '@@ total time: ', end_time - start_time
