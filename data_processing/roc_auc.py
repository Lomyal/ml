#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
#
# Copyright (c) 2015, Inc. All Rights Reserved
#
################################################################################
"""
This module plot ROC and calculate AUC of ROC.

Authors: Wang Shijun
Date:    2015/06/25 09:46:00
"""

import numpy as np
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import datetime
# import pprint as pp

class Everything(object):
    """
    目前所有计算都集中在这里
    """

    def __init__(self, mf, tf):

        # 模型文件名
        self.mf = mf
        # 测试集文件名数组
        self.tf = tf
        # 模型向量维数
        self.dim_num = 0
        # 每个测试集文件的行数（数据条数）
        self.data_num_perfile = []
        # 总测试数据数
        self.data_num = 0

        # 获取模型向量维数
        with open(self.mf) as model:
            self.dim_num = sum(1 for line in model)
        # 获取测试集数据数
        for tf in self.tf:
            with open(tf) as test:
                line_sum = sum(1 for line in test)
                self.data_num_perfile.append(line_sum)
                self.data_num += line_sum
        print self.data_num

        # 模型向量（特征名称到特征取值的映射）
        self.w = dict()
        # 一条测试集数据向量
        self.x = np.zeros(self.dim_num)
        # 全部测试集数据的y构成的向量（AUC计算函数的输入1）
        self.y = np.zeros(self.data_num)
        # 全部测试集数据x输入模型后得到的预测结果构成的向量（AUC计算函数的输入2）
        self.score = np.zeros(self.data_num)

    def construct_base(self):
        """
        构造AUC计算函数的输入
        :return:
        """

        # 构造模型向量w
        with open(self.mf) as model:
            for index, line in enumerate(model):
                w, dummy, feature = line.strip().split('\t')
                self.w[feature] = float(w)

        # 构造x、y、score
        for file_index, tf in enumerate(self.tf):  # 遍历测试集文件
            with open(tf) as test:
                for line_index, line in enumerate(test):  # 遍历文件中的行
                    pre_lines = 0
                    for i in range(file_index):
                        pre_lines += self.data_num_perfile[i]  # 计算该文件之前的数据行数
                    index = pre_lines + line_index  # 计算跨文件的数据行编号
                    if index % 1000000 == 0:
                        print '{time}'.format(time=datetime.datetime.now())
                        print 'processing %d | %.1f%%' % (index, float(index) * 100 / float(self.data_num))
                    feature = line.strip().split(',')  # 获取应该在

                    # 使用优化的方式计算w与x的内积
                    inner_prod_wx = 0.0
                    for fea in feature[2:]:
                        try:
                            inner_prod_wx += self.w[fea.strip()]  # 将出现在该条测试集数据中的特征所对应的位置置为1
                        except:
                            pass  # 忽略测试集中存在，但模型向量w中不存在的特征

                    # 填写y和score中相应于该条测试数据的位置
                    self.y[index] = int(feature[1])
                    self.score[index] = self.calc_score(inner_prod_wx)
        print 'construction complete'

    @staticmethod
    def calc_score(inner_prod_wx):
        """
        对每一条测试集数据，使用逻辑回归模型计算点击率
        :param inner_prod_wx:
        :return:
        """
        exp = np.exp(inner_prod_wx)
        score = exp / (1 + exp)
        return score

    def calc_auc(self):
        """
        计算AUC
        :return:
        """
        auc = sklm.roc_auc_score(self.y, self.score)
        print 'AUC = %f' % auc

        fpr, tpr, thresholds = sklm.roc_curve(self.y, self.score)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('result/roc.png', dpi=200)


if __name__ == '__main__':
    model_file = 'data/originID_model_result_20150624080508.0'
    test_file = ['data/selfmade_test_big']

    time_1 = datetime.datetime.now()
    e = Everything(model_file, test_file)
    time_2 = datetime.datetime.now()
    e.construct_base()
    time_3 = datetime.datetime.now()
    e.calc_auc()
    time_4 = datetime.datetime.now()

    print 'total time: %fs' % (time_4 - time_1).seconds
    print 'class construction time: %fs' % (time_2 - time_1).seconds
    print 'data construction time: %fs' % (time_3 - time_2).seconds
    print 'auc calculation time: %fs' % (time_4 - time_3).seconds
