import copy
import os
import math
import numpy as np
import sys
import csv
from ppo import run_ppo
from evogym import sample_robot, hashable
# import torch.multiprocessing as mp
import utils.my_mp_group as mp
from utils.algo_utils import get_percent_survival_evals, crossover, mutate, TerminationCondition, Structure, structure_to_genome

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))


def variation(parent, par_fit, record, pop_size):
    par_size = len(parent)
    off_structure = []
    off_connection = []
    while len(off_structure) < pop_size:
        # 二元锦标赛选择
        r = np.random.choice(np.arange(0, par_size), 2, replace=False)
        r1 = r[0]
        r2 = r[1]
        p1 = copy.deepcopy(parent[r1])
        p2 = copy.deepcopy(parent[r2])
        cro_off = crossover(p1, p2)
        for co in cro_off:
            tmp_child = mutate(co, mutation_rate = 0.1, num_attempts=30)
            if tmp_child != None and (hashable(tmp_child[0]) not in record):
                off_structure.append(tmp_child[0])
                off_connection.append(tmp_child[1])
    return off_structure, off_connection


## 获得模型的输出，训练模型用
## reward的类型是字典，把它转换成数组
def get_output(reward):
    output = np.zeros(len(reward))
    for i in reward.keys():
        output[int(i)] = reward[i]
    return output


## 根据real_fits从大到小，只拿出前pop_size个个体
def maintain_pop(pop_genome, real_fits, pop_size):
    parent = []
    parent_fits = []
    ## 当前排序结果是从小到大
    sorted_id = np.argsort(real_fits)
    i = len(sorted_id) - 1
    while len(parent) < pop_size:
        sid = sorted_id[i]
        parent.append(pop_genome[sid])
        parent_fits.append(real_fits[sid])
        i = i - 1
    return parent, parent_fits


def run_standard_ga(experiment_name, structure_shape, pop_size, max_evaluations, train_iters, num_cores, env_name):
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    ## 强化学习的终止条件
    tc = TerminationCondition(train_iters)
    ## 迭代次数
    gen = 0
    ## 优化中所有个体的哈希表
    population_hashes = {}
    ## 当前评价次数
    num_evaluations = 0

    ## 随机初始化初代种群
    # 初始化种群
    pop_structure = []
    pop_connection = []
    for i in range(pop_size):
        structure, connection = sample_robot(structure_shape)
        while (hashable(structure) in population_hashes):
            structure, connection = sample_robot(structure_shape)
        pop_structure.append(structure)
        pop_connection.append(connection)
        population_hashes[hashable(structure)] = True

    # 创建文件夹
    gen_path = os.path.join(home_path, 'generation_'+str(gen))
    controler_path = os.path.join(gen_path, 'controler')
    structure_path = os.path.join(gen_path, 'structure')

    try:
        os.makedirs(controler_path)
    except:
        pass

    try:
        os.makedirs(structure_path)
    except:
        pass

    # 评价初代种群
    group = mp.Group()
    for i in range(len(pop_structure)):
        ppo_args = ((pop_structure[i], pop_connection[i]), tc, (controler_path, i), env_name)
        group.add_job(run_ppo, ppo_args)
        temp_path = os.path.join(structure_path, str(i))
        np.savez(temp_path, pop_structure[i], pop_connection[i])
    group.run_jobs(num_cores)
    rewards = group.reward
    pop_fits = get_output(rewards)

    num_evaluations = num_evaluations + len(pop_fits)

    ## 写入所有真实评估的个体的适应度值，随时保存结果
    optima_path = os.path.join(home_path, 'generations.csv')
    with open(optima_path, 'w', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        for pfi in range(len(pop_fits)):
            writer.writerow([pfi, pop_fits[pfi]])

    while num_evaluations < max_evaluations:

        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))

        if max_evaluations - num_evaluations < pop_size-num_survivors:
            new_off_size = max_evaluations - num_evaluations
        else:
            new_off_size = pop_size-num_survivors
        # 进入下一代
        gen = gen + 1
        # 创建保存数据的文件夹
        gen_path = os.path.join(home_path, 'generation_' + str(gen))
        controler_path = os.path.join(gen_path, 'controller')
        structure_path = os.path.join(gen_path, 'structure')
        try:
            os.makedirs(controler_path)
        except:
            pass

        try:
            os.makedirs(structure_path)
        except:
            pass

        # 产生子代
        off_structure, off_connection = variation(pop_structure, pop_fits, population_hashes, new_off_size)
        off_size = len(off_structure)

        group = mp.Group(rewards, num_evaluations)
        for j in range(len(off_structure)):
            ppo_args = ((off_structure[j], off_connection[j]), tc, (controler_path, num_evaluations+j+1), env_name)
            group.add_job(run_ppo, ppo_args)
            temp_path = os.path.join(structure_path, str(num_evaluations+j+1))
            np.savez(temp_path, off_structure[j], off_connection[j])
        group.run_jobs(num_cores)
        rewards = group.reward
        all_fits = get_output(rewards)

        off_fits = all_fits[num_evaluations:num_evaluations+off_size]
        # 更新评价次数
        num_evaluations = num_evaluations+off_size

        pop_structure, pop_fits = maintain_pop(pop_structure+off_structure, list(pop_fits)+list(off_fits), num_survivors)

        ## 写入所有真实评估的个体的适应度值，随时保存结果
        optima_path = os.path.join(home_path, 'generations.csv')
        with open(optima_path, 'w', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            for pfi in range(len(all_fits)):
                writer.writerow([pfi, all_fits[pfi]])
