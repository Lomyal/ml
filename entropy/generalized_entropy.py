# -*- coding:utf-8 -*-

from __future__ import division
from math import *
import numpy as np


def get_joint_dist(var_num, dist):
    """
    计算联合分布矩阵（维度为任意1~n）
    :param var_num: 联合分布维度大小
    :param dist: 每种取值组合的个数
    :return:
    """
    v = []  # 记录每一个随机变量每种取值对应在联合分布多维矩阵中的位置
    n = []  # 每种随机变量取值个数统计
    for i in range(var_num):
        v.append({})
        n.append(0)

    # 统计每个随机变量的取值个数，并将取值映射到多维矩阵中的位置上
    for item in dist:
        for i in range(var_num):
            if item[i] not in v[i]:
                v[i][item[i]] = n[i]
                n[i] += 1

    # 用0.0初始化多维矩阵
    arg = ()
    for num in n:
        arg += (num,)
    jd = np.zeros(arg, np.float64)

    # 统计每种组合取值的个数
    total = 0
    for item in dist:
        total += item[-1]
        prob = jd
        for i in range(var_num):
            index = v[i][item[i]]
            if i != var_num - 1:
                prob = prob[index]
            else:
                prob[index] += item[-1]

    # 归一化
    jd /= total
    return jd, v, n


def get_entropy(dist):
    """
    计算经典熵
    :param var_num:
    :param dist:
    :return:
    """
    var_num = len(dist[0]) - 1

    def unit_calc(p):
        if p != 0:
            return -p * log(p) / log(2)
        else:
            return 0.0

    joint_dist, _, _ = get_joint_dist(var_num, dist)
    ent = 0.0

    # 遍历联合分布中每一个概率
    for prob in joint_dist.flat:
        ent += unit_calc(prob)
    return ent


def get_similarity(varis):
    """
    计算随机变量取值的相似度
    :param varis:
    :return:
    """

    def jaccard_sim(s1, s2):
        return len(s1 & s2) / len(s1 | s2)

    var_num = len(varis)
    sims = []

    for i in range(var_num):
        val_num = len(varis[i])
        sim = np.zeros((val_num, val_num), np.float64)
        for row in varis[i]:
            for col in varis[i]:
                sim[varis[i][row], varis[i][col]] = jaccard_sim(set(row), set(col))
        sims.append(sim)

    return sims


def get_generalized_entropy(dist):
    """
    广义熵
    :param dist:
    :return:
    """
    var_num = len(dist[0]) - 1
    joint_dist, variables, _ = get_joint_dist(var_num, dist)
    similarity = get_similarity(variables)

    def calc_log_content(k, cont, names, fixed_names):
        """
        递归计算log中的内容
        :param k:
        :param cont:
        :param names:
        :param fixed_names:
        :return:
        """
        if k == var_num:
            prob = joint_dist
            for i in range(var_num):
                index = variables[i][names[i]]
                prob = prob[index]
            sim = 1
            for i in range(var_num):
                index1 = variables[i][fixed_names[i]]
                index2 = variables[i][names[i]]
                sim *= similarity[i][index1][index2]
            return cont + prob * sim

        for key in variables[k]:
            names.append(key)
            cont = calc_log_content(k + 1, cont, names, fixed_names)
            names.pop()

        return cont

    def calc_ge(k, ent, names):
        """
        递归计算generalized entropy
        :param k:
        :param ent:
        :param names:
        :return:
        """
        if k == var_num:
            log_cont = calc_log_content(0, 0.0, [], names)
            prob = joint_dist
            for i in range(var_num):
                prob = prob[variables[i][names[i]]]
            if log_cont == 0 and prob == 0:
                return ent
            return ent - prob * log(log_cont) / log(2)

        for key in variables[k]:
            names.append(key)
            ent = calc_ge(k + 1, ent, names)
            names.pop()

        return ent

    return calc_ge(0, 0.0, [])


def get_con_gen_ent(dist, cond_dimen_list):
    """
    条件广义熵
    :param dist:
    :return:
    """
    var_num = len(dist[0]) - 1
    joint_dist, variables, _ = get_joint_dist(var_num, dist)
    similarity = get_similarity(variables)

    # 对没有出现在条件中的维度求和
    joint_dist_cond = joint_dist
    for i in reversed(range(var_num)):
        if i not in cond_dimen_list:
            joint_dist_cond = joint_dist_cond.sum(axis=i)

    def calc_log_content_up(k, cont, names, fixed_names):
        """
        递归计算log中的分子
        :param k:
        :param cont:
        :param names:
        :param fixed_names:
        :return:
        """
        if k == var_num:
            prob = joint_dist
            for i in range(var_num):
                index = variables[i][names[i]]
                prob = prob[index]
            sim = 1
            for i in range(var_num):
                index1 = variables[i][fixed_names[i]]
                index2 = variables[i][names[i]]
                sim *= similarity[i][index1][index2]
            return cont + prob * sim

        for key in variables[k]:
            names.append(key)
            cont = calc_log_content_up(k + 1, cont, names, fixed_names)
            names.pop()

        return cont

    def calc_log_content_dn(k, cont, names, fixed_names):
        """
        递归计算log中的分母
        :param k:
        :param cont:
        :param names:
        :param fixed_names:
        :return:
        """
        if k == var_num:
            prob = joint_dist_cond
            skip = 0
            for i in range(var_num):
                if i in cond_dimen_list:
                    index = variables[i][names[i - skip]]
                    prob = prob[index]
                else:
                    skip += 1
            sim = 1
            skip = 0
            for i in range(var_num):
                if i in cond_dimen_list:
                    index1 = variables[i][fixed_names[i]]
                    index2 = variables[i][names[i - skip]]
                    sim *= similarity[i][index1][index2]
                else:
                    skip += 1
            return cont + prob * sim

        if k in cond_dimen_list:
            for key in variables[k]:
                names.append(key)
                cont = calc_log_content_dn(k + 1, cont, names, fixed_names)
                names.pop()
        else:
            cont = calc_log_content_dn(k + 1, cont, names, fixed_names)

        return cont

    def calc_cge(k, ent, names):
        """
        递归计算conditional generalized entropy
        :param k:
        :param ent:
        :param names:
        :return:
        """
        if k == var_num:
            log_cont_up = calc_log_content_up(0, 0.0, [], names)
            log_cont_dn = calc_log_content_dn(0, 0.0, [], names)
            if log_cont_dn == 0:  # 应对X或Y中没有随机变量的情况
                log_cont = 1
            else:
                log_cont = log_cont_up / log_cont_dn
            prob = joint_dist
            for i in range(var_num):
                prob = prob[variables[i][names[i]]]
            if log_cont == 0 and prob == 0:
                return ent
            return ent - prob * log(log_cont) / log(2)

        for key in variables[k]:
            names.append(key)
            ent = calc_cge(k + 1, ent, names)
            names.pop()

        return ent

    return calc_cge(0, 0.0, [])


def get_mutual_conflict(dist, cond_dimen_list):
    """
    互分歧度
    :param dist:
    :param cond_dimen_list: 在list中的随机变量是一边，不在list中的随机变量是另一边
    :return:
    """
    var_num = len(dist[0]) - 1
    joint_dist, variables, _ = get_joint_dist(var_num, dist)
    similarity = get_similarity(variables)

    # 对出现在条件中的维度求和
    joint_dist_cond_1 = joint_dist
    for i in reversed(range(var_num)):
        if i in cond_dimen_list:
            joint_dist_cond_1 = joint_dist_cond_1.sum(axis=i)

    # 对没有出现在条件中的维度求和
    joint_dist_cond_2 = joint_dist
    for i in reversed(range(var_num)):
        if i not in cond_dimen_list:
            joint_dist_cond_2 = joint_dist_cond_2.sum(axis=i)

    def calc_log_content_up(k, cont, names, fixed_names):
        """
        递归计算log中的分子
        :param k:
        :param cont:
        :param names:
        :param fixed_names:
        :return:
        """
        if k == var_num:
            prob = joint_dist
            for i in range(var_num):
                index = variables[i][names[i]]
                prob = prob[index]
            sim = 1
            for i in range(var_num):
                index1 = variables[i][fixed_names[i]]
                index2 = variables[i][names[i]]
                sim *= similarity[i][index1][index2]
            return cont + prob * sim

        for key in variables[k]:
            names.append(key)
            cont = calc_log_content_up(k + 1, cont, names, fixed_names)
            names.pop()

        return cont

    def calc_log_content_dn_1(k, cont, names, fixed_names):
        """
        递归计算log中的分母1
        :param k:
        :param cont:
        :param names:
        :param fixed_names:
        :return:
        """
        if k == var_num:
            prob = joint_dist_cond_1
            skip = 0
            for i in range(var_num):
                if i not in cond_dimen_list:
                    index = variables[i][names[i - skip]]
                    prob = prob[index]
                else:
                    skip += 1
            sim = 1
            skip = 0
            for i in range(var_num):
                if i not in cond_dimen_list:
                    index1 = variables[i][fixed_names[i]]
                    index2 = variables[i][names[i - skip]]
                    sim *= similarity[i][index1][index2]
                else:
                    skip += 1
            return cont + prob * sim

        if k not in cond_dimen_list:
            for key in variables[k]:
                names.append(key)
                cont = calc_log_content_dn_1(k + 1, cont, names, fixed_names)
                names.pop()
        else:
            cont = calc_log_content_dn_1(k + 1, cont, names, fixed_names)

        return cont

    def calc_log_content_dn_2(k, cont, names, fixed_names):
        """
        递归计算log中的分母2
        :param k:
        :param cont:
        :param names:
        :param fixed_names:
        :return:
        """
        if k == var_num:
            prob = joint_dist_cond_2
            skip = 0
            for i in range(var_num):
                if i in cond_dimen_list:
                    index = variables[i][names[i - skip]]
                    prob = prob[index]
                else:
                    skip += 1
            sim = 1
            skip = 0
            for i in range(var_num):
                if i in cond_dimen_list:
                    index1 = variables[i][fixed_names[i]]
                    index2 = variables[i][names[i - skip]]
                    sim *= similarity[i][index1][index2]
                else:
                    skip += 1
            return cont + prob * sim

        if k in cond_dimen_list:
            for key in variables[k]:
                names.append(key)
                cont = calc_log_content_dn_2(k + 1, cont, names, fixed_names)
                names.pop()
        else:
            cont = calc_log_content_dn_2(k + 1, cont, names, fixed_names)

        return cont

    def calc_mc(k, ent, names):
        """
        递归计算conditional generalized entropy
        :param k:
        :param ent:
        :param names:
        :return:
        """
        if k == var_num:
            log_cont_up = calc_log_content_up(0, 0.0, [], names)
            log_cont_dn_1 = calc_log_content_dn_1(0, 0.0, [], names)
            log_cont_dn_2 = calc_log_content_dn_2(0, 0.0, [], names)
            if log_cont_dn_1 == 0 or log_cont_dn_2 == 0:  # 应对X或Y中没有随机变量的情况
                log_cont = 1
            else:
                log_cont = log_cont_up / log_cont_dn_1 / log_cont_dn_2
            prob = joint_dist
            for i in range(var_num):
                prob = prob[variables[i][names[i]]]
            if log_cont == 0 and prob == 0:
                return ent
            return ent + prob * log(log_cont) / log(2)

        for key in variables[k]:
            names.append(key)
            ent = calc_mc(k + 1, ent, names)
            names.pop()

        return ent

    return calc_mc(0, 0.0, [])


def get_vars(dist, dims):
    def select_dims(tup):
        ret = ()
        for dim in dims:
            ret += (tup[dim],)
        ret += (tup[-1],)
        return ret

    ok = map(select_dims, dist)
    return ok


def print_entropy(ucs):
    print '== print_entropy =='
    for i, uc in enumerate(ucs):
        print '{no}  {e}  {ge}  {title}'.format(no='%3d' % (i + 1),
                                                e='%.3f' % get_entropy(uc),
                                                ge='%.3f' % get_generalized_entropy(uc),
                                                title=str(uc))


def print_conditional_entropy(ucs, cond_list):
    print '== print_conditional_entropy =='
    for i, uc in enumerate(ucs):
        print '{no}  {cge}  {title}'.format(no='%3d' % (i + 1),
                                            cge='%.3f' % get_con_gen_ent(uc, cond_list),
                                            title=str(uc))


def print_mutual_conflict(ucs, cond_list):
    print '== print_mutual_conflict =='
    for i, uc in enumerate(ucs):
        print '{no}  {cge}  {title}'.format(no='%3d' % (i + 1),
                                            cge='%.3f' % get_mutual_conflict(uc, cond_list),
                                            title=str(uc))


def print_all(ucs, cond_list):
    # print ' no  jge    sge    cge    mc     title'
    all_list = range(len(ucs[0][0]) - 1)
    cond_list_comp = list(set(all_list).difference(set(cond_list)))
    print ''
    print '========================'
    print ' ALL DIMS: {}'.format(str(all_list))
    print '   X DIMS: {}'.format(str(cond_list_comp))
    print '   Y DIMS: {}'.format(str(cond_list))
    print '------------------------'
    print ' NO   K(X,Y)  K(X)    K(Y)    K(X|Y)  K(Y|X)  C(X;Y)  TITLE'
    for i, uc in enumerate(ucs):
        line = '{no}   {ge}   {sge}   {sge2}   {cge}   {cge2}   {mc}   {title}'
        print line.format(no='%3d' % (i + 1),
                          ge='%.3f' % get_generalized_entropy(uc),
                          sge='%.3f' % get_generalized_entropy(get_vars(uc, cond_list_comp)),
                          sge2='%.3f' % get_generalized_entropy(get_vars(uc, cond_list)),
                          cge='%.3f' % get_con_gen_ent(uc, cond_list),
                          cge2='%.3f' % get_con_gen_ent(uc, cond_list_comp),
                          mc='%.3f' % get_mutual_conflict(uc, cond_list),
                          title=str(uc))


if __name__ == '__main__':

    # user_choices = [
    #     [('a', 'A', 50), ('ab', 'A', 10), ('ab', 'AB', 20), ('b', 'AB', 20)],
    #     [('a', 'A', 30), ('a', 'AB', 20), ('ab', 'A', 18), ('ab', 'AB', 12), ('b', 'A', 12), ('b', 'AB', 8)],
    # ]
    # print_all(user_choices, [1])
    # print_all(user_choices, [])
    #
    # user_choices = [
    #     [('a', 'A', '1', 50), ('ab', 'A', '2', 10), ('ab', 'AB', '12', 20), ('b', 'AB', '1', 20)],
    #     [('a', 'A', '1', 50), ('b', 'A', '1', 10), ('c', 'A', '1', 20), ('d', 'A', '1', 20)],
    #     # [('a', 'A', 30), ('a', 'AB', 20), ('ab', 'A', 18), ('ab', 'AB', 12), ('b', 'A', 12), ('b', 'AB', 8)],
    # ]
    # print_all(user_choices, [])

    user_choices = [
        [('a', 'A', 1), ('b', 'B', 1)],
        [('Salesman', '1', 1), ('Seller', '2', 1)],
    ]
    print_all(user_choices, [1])
    # print_all(user_choices, [])

    # user_choices = [
    #     [('a', 'A', '1', '#', 50), ('b', 'A', '1', '#', 10), ('c', 'A', '1', '#', 20), ('d', 'A', '1', '#', 20)],
    #     [('a', 'A', '1', '#', 50), ('ab', 'A', '1', '#', 10), ('bc', 'A', '1', '#', 20), ('ad', 'A', '1', '#', 20)],
    #     [('a', 'A', '1', '#', 50), ('a', 'A', '1', '#', 10), ('a', 'A', '1', '#', 20), ('ab', 'A', '1', '#', 20)],
    # ]
    # print_all(user_choices, [])
    # print_all(user_choices, [0])
    # print_all(user_choices, [1])
    # print_all(user_choices, [2])
    # print_all(user_choices, [3])

