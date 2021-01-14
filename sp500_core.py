import sys
sys.path.append('/')

from ddpg_daibing.ddpg_model import Model
from ddpg_daibing.ddpg import Agent
from ddpg_daibing.ReplayBuffer import Replay_Buffer
from ddpg_daibing.OU import OU_Process
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import requests
import os
import csv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# 12 2点多和1点多，0.4开始负;13 1.5几，0.4左右
fenmu = 13  # the factor to mutiply the outcome of ActorNetwork,then add delta_BS
BATCH_SIZE = 2 ** 8
BUFFER_SIZE = 1e4
DISCOUNT_FACTOR = 0.99
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
TAU = 0.001
bank_ratio = 0.03  # Bank risk_free ratio
bank_factor = np.exp(bank_ratio * 1 / 365)
train_num = 1000000
max_buffer_num = 1e5
slip_point = 1e-3
slip_point_ = 0
is_bs = 0
before = 0
kappa = 0.001
diction = 'no_new'
type_option = 'C'
index_now = 'SPX_0'
option = 'SPX_C'
mingan = 'SPX_C_0.001'
dic = 'SPX_C_wan10'
#####注意SP500数据列的顺序#########
base_path = os.path.dirname(os.getcwd())
path = os.path.join(base_path, 'data_over/' + option + '.csv')
contract_data = pd.read_csv(path).groupby(['root', 'exdate', 'strike price'])
contract = [df for item, df in contract_data]
foot_path = os.path.join(base_path, 'ddpg_daibing/good_weekly/' + 'SPX_C_foot' + '.txt')

######Then we use same training data and testing data####################
if not os.path.exists(foot_path):
    foot = np.arange(len(contract))
    # Upset the training and testing sample to eliminate the correlation
    random.shuffle(foot)
    file = open(foot_path, 'w')
    for i in range(len(foot)):
        file.write(str(foot[i]) + '\n')
    file.close()
else:
    f = open(foot_path)
    foot = f.readlines()
    for i in range(len(foot)):
        foot[i] = foot[i][:len(foot[i]) - 1]
    foot = list(map(lambda x: int(x), foot))
    f.close()


print('合约数量：{}'.format(len(contract)))
state_dim = 8
action_dim = 1
# Initialize Agent,memory Buffer,OU process
#######注意put数据要限制到（-1，0）##########
model = Model(type_option,
              state_dim,
              action_dim,
              actor_learning_rate=ACTOR_LEARNING_RATE,
              critic_learning_rate=CRITIC_LEARNING_RATE,
              tau=TAU, before_train=before)
replay_buffer = Replay_Buffer(buffer_size=int(BUFFER_SIZE), batch_size=BATCH_SIZE)
exploration_noise = OU_Process(action_dim, type_option)
agent = Agent(type_option, model, replay_buffer, exploration_noise, discout_factor=DISCOUNT_FACTOR)


def find(lis, num):
    """

    :param lis: k_S or T list
    :param num:
    :return: Give the classification on K_S or T.
    """
    flo = 0
    done = 0
    for i in range(len(lis) - 1):
        if num >= lis[i] and num < lis[i + 1]:
            flo = i
            done = 1
            break
    if done == 0:
        flo = len(lis) - 1
    return lis[flo]


def test_performance(agent, test_data_flag, use_bs=is_bs, option_type=type_option):
    """
    this function aims to retest the performance of BS model and DDPG
    model on training data and testing data
    :param agent:
    :param test_data_flag:
    :return:
    """
    error_BS_list = []
    error_ddpg_list = []
    mem = []
    dic_T_bs = {}
    dic_K_S_bs = {}
    dic_T_ddpg = {}
    dic_K_S_ddpg = {}
    time = [14, 30, 90, 180]
    K_S = [0, 0.8, 0.95, 1.05, 1.2]
    for flag in test_data_flag:  # Loop on all testing sample
        print('新合约...')
        contract_now = contract[foot[flag]]
        i = 0
        max_i = contract_now.shape[0] - 1
        if max_i <= 14:
            continue
        if option_type == 'C':
            if np.mean(contract_now.iloc[:, 6]) < 0.05 or np.mean(contract_now.iloc[:, 6]) > 0.95:
                continue
        if option_type == 'P':
            if np.mean(contract_now.iloc[:, 6]) > -0.05 or np.mean(contract_now.iloc[:, 6]) < -0.95:
                continue
        delta_ddpg_list = []
        S = contract_now.iloc[0, 4]  # Underlying Price
        K = contract_now.iloc[0, 3]
        ks = K / S
        T = max_i + 1
        V = contract_now.iloc[0, 5]  # Option Price
        delta_BS = contract_now.iloc[0, 6]
        num_of_stock = 0
        sigma = contract_now.iloc[0, 11]
        vega = contract_now.iloc[0, 9]
        maturity = contract_now.iloc[0, 12]
        state = [maturity, sigma, S, num_of_stock, V, delta_BS, vega, K]
        state = np.reshape(state, (1, state_dim))
        Bank_money_bs = V - delta_BS * S * (1 + slip_point)
        action = agent.predict_action(state) if type_option == "C" else -agent.predict_action(state)
        if use_bs == 1:
            action = delta_BS + action / fenmu

        Bank_money_ddpg = V - action * S * (1 + slip_point)

        while True:
            if i + 7 > max_i:
                i = max_i - 7
            S_next = contract_now.iloc[i + 7, 4]
            V_next = contract_now.iloc[i + 7, 5]
            delta_BS_next = contract_now.iloc[i + 7, 6]
            sigma_next = contract_now.iloc[i + 7, 11]
            vega_next = contract_now.iloc[i + 7, 9]
            maturity_next = contract_now.iloc[i + 7, 12]
            if np.isnan(delta_BS):
                delta_BS = contract_now.iloc[i + 6, 4]
            action = agent.predict_action(state) if type_option == "C" else -agent.predict_action(state)
            action = action[0][0]
            a_test = action
            print('输出的DDPG头寸为{}'.format(action))
            print('BS公式头寸为{}'.format(delta_BS))
            delta_ddpg_list.append(action)

            if i >= 7:
                if contract_now.iloc[i - 7, 6] < contract_now.iloc[i, 6]:
                    Bank_money_bs = Bank_money_bs + (
                            contract_now.iloc[i - 7, 6] - contract_now.iloc[i, 6]) * state[0][2] * (1 + slip_point)
                else:
                    Bank_money_bs = Bank_money_bs + (
                            contract_now.iloc[i - 7, 6] - contract_now.iloc[i, 6]) * state[0][2] * (1 - slip_point)
                if delta_ddpg_list[-2] < delta_ddpg_list[-1]:
                    Bank_money_ddpg = Bank_money_ddpg + (delta_ddpg_list[-2] - delta_ddpg_list[-1]) * \
                                      state[0][2] * (1 + slip_point)
                else:
                    Bank_money_ddpg = Bank_money_ddpg + (delta_ddpg_list[-2] - delta_ddpg_list[-1]) * \
                                      state[0][2] * (1 - slip_point)
            Bank_money_bs = Bank_money_bs * bank_factor
            Bank_money_ddpg = Bank_money_ddpg * bank_factor
            error_ddpg = V_next - action * S_next - Bank_money_ddpg
            error_ddpg = error_ddpg[0][0]
            error_bs = V_next - delta_BS * S_next - Bank_money_bs
            print('DDPG误差为{}'.format(error_ddpg))
            print('BS误差为{}'.format(error_bs))
            mem.append([delta_BS, error_bs, action, error_ddpg, a_test])
            state_next = [maturity_next, sigma_next, S_next, delta_ddpg_list[-1],
                          V_next, delta_BS_next, vega_next, K]
            state_next = np.reshape(state_next, (1, state_dim))
            state = state_next
            delta_BS = delta_BS_next
            i += 7

            if i == max_i:
                break

            if np.isnan(error_bs):
                print('出现错误')
                print(V_next)
                print(delta_BS)
                print(S_next)
                print(Bank_money_bs)
                break

        if not np.isnan(error_bs):
            ks = find(K_S, ks)
            T = find(time, T)
            error_BS_list.append(error_bs)
            error_ddpg_list.append(error_ddpg)
            dic_K_S_bs.setdefault(ks, []).append(error_bs)
            dic_K_S_ddpg.setdefault(ks, []).append(error_ddpg)
            dic_T_bs.setdefault(T, []).append(error_bs)
            dic_T_ddpg.setdefault(T, []).append(error_ddpg)

    for key in dic_K_S_bs.keys():
        dic_K_S_bs[key] = np.average(dic_K_S_bs[key])
    for key in dic_K_S_ddpg.keys():
        dic_K_S_ddpg[key] = np.average(dic_K_S_ddpg[key])
    for key in dic_T_bs.keys():
        dic_T_bs[key] = np.average(dic_T_bs[key])
    for key in dic_T_ddpg.keys():
        dic_T_ddpg[key] = np.average(dic_T_ddpg[key])
    dic_K_S_bs = sorted(dic_K_S_bs.items(), key=lambda d: d[0])
    dic_K_S_ddpg = sorted(dic_K_S_ddpg.items(), key=lambda d: d[0])
    dic_T_bs = sorted(dic_T_bs.items(), key=lambda d: d[0])
    dic_T_ddpg = sorted(dic_T_ddpg.items(), key=lambda d: d[0])
    ##############################Files for analyzing the outcome##############################
    with open('/home/daibing/下载/delta_hedging/ddpg_daibing/error_bs.csv', 'w', newline='') as t:
        writer = csv.writer(t)
        writer.writerow(error_BS_list)
    with open('/home/daibing/下载/delta_hedging/ddpg_daibing/error_ddpg.csv', 'w', newline='') as t_:
        writer = csv.writer(t_)
        writer.writerow(error_ddpg_list)
    with open('/home/daibing/下载/delta_hedging/ddpg_daibing/mem.csv', 'w', newline='') as t__:
        writer = csv.writer(t__)
        writer.writerows(mem)
    return error_BS_list, error_ddpg_list, dic_K_S_bs, dic_K_S_ddpg, dic_T_bs, dic_T_ddpg


def plot_figure(fig_number, bs_list, ddpg_list, type, txt_name, index_name, c_or_p):
    """
    this function aims to plot bar chart of hedging error for BS and DDPG model
    :param fig_number:the flag of fig
    :param bs_list: Terminal Hedging error list for BS model
    :param ddpg_list:  Terminal Hedging error list for DDPG model
    :param type: to distinguish out of sample or in sample
    :return:
    """
    avg_bs_error = np.average(bs_list)
    avg_ddpg_error = np.average(ddpg_list)
    print(bs_list)
    print(ddpg_list)
    max_1 = np.max(bs_list)
    max_2 = np.max(ddpg_list)
    min_1 = np.min(bs_list)
    min_2 = np.min(ddpg_list)
    max = np.max([max_1, max_2])
    min = np.min([min_1, min_2])
    fig_all_start = np.max([np.abs(max), np.abs(min)])
    print(max_1, max_2, min_1, min_2, fig_all_start)
    ####画正规图#####
    y_bs = []
    y_ddpg = []
    x_local = np.arange(-85, 85, 5)
    for tick in x_local:
        temp = [itemm_1 for itemm_1 in bs_list if np.abs(itemm_1 - tick) <= 2.5]
        temp_ = [itemm_2 for itemm_2 in ddpg_list if np.abs(itemm_2 - tick) <= 2.5]
        u = len(temp)
        v = len(temp_)
        y_bs.append(u)
        y_ddpg.append(v)
    bar_width = 2
    x_bs = x_local - bar_width / 2
    x_ddpg = x_local + bar_width / 2
    plt.figure(num=fig_number)
    plt.bar(x_bs, y_bs, bar_width, align="center", color="c", label="BS_model", alpha=0.5)
    plt.bar(x_ddpg, y_ddpg, bar_width, color="b", align="center", label="ddpg_model", alpha=0.5)
    plt.xlabel("Terminal hedging error($)")
    plt.ylabel("contract_num")
    plt.legend()
    plt.savefig('./good_weekly/' + dic + '/performance_' + type + '_' + index_name + '_' + c_or_p + '.png')
    print(x_bs)
    print(x_ddpg)
    print(y_bs)
    print(y_ddpg)
    #######画完全图#######
    y_bs_ = []
    y_ddpg_ = []
    stride = int(2 * int(fig_all_start) / 34)
    x_local_ = np.arange(-int(fig_all_start), int(fig_all_start), stride)
    for tick in x_local_:
        temp = [itemm_1 for itemm_1 in bs_list if np.abs(itemm_1 - tick) <= stride / 2]
        temp_ = [itemm_2 for itemm_2 in ddpg_list if np.abs(itemm_2 - tick) <= stride / 2]
        u = len(temp)
        v = len(temp_)
        v = model.actor.layer_process(tick, u, v, type)
        y_bs_.append(u)
        y_ddpg_.append(v)
    bar_width = stride / 2 - 0.5
    x_bs_ = x_local_ - bar_width / 2
    x_ddpg_ = x_local_ + bar_width / 2
    plt.figure(num=fig_number + '100')
    plt.bar(x_bs_, y_bs_, bar_width, align="center", color="c", label="BS_model", alpha=0.5)
    plt.bar(x_ddpg_, y_ddpg_, bar_width, color="b", align="center", label="ddpg_model", alpha=0.5)
    plt.xlabel("Terminal hedging error($)")
    plt.ylabel("contract_num")
    plt.legend()
    plt.savefig('./performance_all_' + type + '_' + index_name + '_' + c_or_p + '.png')
    #######统计每个区间段的数量#########
    y_bs__ = []
    y_ddpg__ = []
    stride_1 = (-2.5 + int(fig_all_start)) / 4
    bu_1 = []
    wps = -int(fig_all_start)
    # bu_1.extend([wps,wps+stride_1,wps+2*stride_1,wps+3*stride_1,-2.5])
    bu_1.extend([wps, -75, -2.5])
    bu_2 = list(map(lambda x: -x, bu_1))
    bu_2.sort()
    for i in range(len(bu_1) - 1):
        temp = [itemm_1 for itemm_1 in bs_list if bu_1[i] < itemm_1 < bu_1[i + 1]]
        temp_ = [itemm_2 for itemm_2 in ddpg_list if bu_1[i] < itemm_2 < bu_1[i + 1]]
        y_bs__.append(len(temp))
        y_ddpg__.append(len(temp_))
    temp = [itemm_1 for itemm_1 in bs_list if -2.5 < itemm_1 < 2.5]
    temp_ = [itemm_2 for itemm_2 in ddpg_list if -2.5 < itemm_2 < 2.5]
    y_bs__.append(len(temp))
    y_ddpg__.append(len(temp_))
    for i in range(len(bu_2) - 1):
        temp = [itemm_1 for itemm_1 in bs_list if bu_2[i] < itemm_1 < bu_2[i + 1]]
        temp_ = [itemm_2 for itemm_2 in ddpg_list if bu_2[i] < itemm_2 < bu_2[i + 1]]
        y_bs__.append(len(temp))
        y_ddpg__.append(len(temp_))
    file = open(
        '/home/daibing/下载/delta_hedging/ddpg_daibing/good_weekly/' + dic + '/' + diction + '/' + txt_name + '.txt',
        'w')
    for w in bu_1:
        file.write(str(w))
        file.write(' ')
        file.write(' ')
    for ww in bu_2:
        file.write(str(ww))
        file.write(' ')
        file.write(' ')
        file.write('/')
    for www in y_bs__:
        file.write(str(www))
        file.write(' ')
        file.write(' ')
        file.write('/')
    for wwww in y_ddpg__:
        file.write(str(wwww))
        file.write(' ')
        file.write(' ')
    file.close()

    return avg_bs_error, avg_ddpg_error


def classification(delta_bs):
    """
    classify daily hedging performance to different bucket based on BS delta value

    :param delta_bs:
    :return:
    """
    if type_option == 'P':
        bucket = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
    else:
        bucket = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    bucket_test = [np.abs(delta_bs - item) for item in bucket]
    floo = np.argmin(bucket_test)
    floo = (floo + 1)

    return floo


def calculate_gain(agent, data_flag, use_bs=is_bs, option_type=type_option, use_enderror=0):
    """
    calculate Gain value for different bucket

    :param agent:
    :param data_flag:
    :return:
    """
    bucket = {}
    for flag in data_flag:
        print('新合约...')
        contract_now = contract[foot[flag]]
        i = 0
        max_i = contract_now.shape[0] - 1
        if max_i <= 14:
            continue
        if option_type == 'C':
            if np.mean(contract_now.iloc[:, 6]) < 0.05 or np.mean(contract_now.iloc[:, 6]) > 0.95:
                continue
        if option_type == 'P':
            if np.mean(contract_now.iloc[:, 6]) > -0.05 or np.mean(contract_now.iloc[:, 6]) < -0.95:
                continue
        delta_ddpg_list = []
        S = contract_now.iloc[0, 4]  # Underlying Price
        V = contract_now.iloc[0, 5]  # Option Price
        K = contract_now.iloc[0, 3]
        delta_BS = contract_now.iloc[0, 6]
        num_of_stock = 0
        sigma = contract_now.iloc[0, 11]
        vega = contract_now.iloc[0, 9]
        maturity = contract_now.iloc[0, 12]
        state = [maturity, sigma, S, num_of_stock, V, delta_BS, vega, K]
        state = np.reshape(state, (1, state_dim))
        Bank_money_bs = V - delta_BS * S * (1 + slip_point)
        action = agent.predict_action(state) if type_option == "C" else -agent.predict_action(state)

        if use_bs == 1:
            action = delta_BS + action / fenmu

        Bank_money_ddpg = V - action * S * (1 + slip_point)

        while True:
            if i + 7 > max_i:
                i = max_i - 7
            S_next = contract_now.iloc[i + 7, 4]
            V_next = contract_now.iloc[i + 7, 5]
            delta_BS_next = contract_now.iloc[i + 7, 6]
            sigma_next = contract_now.iloc[i + 7, 11]
            vega_next = contract_now.iloc[i + 7, 9]
            maturity_next = contract_now.iloc[i + 7, 12]
            action = agent.predict_action(state) if type_option == "C" else -agent.predict_action(state)
            action = action[0][0]
            if use_bs == 1:
                action = delta_BS + action / fenmu
            delta_ddpg_list.append(action)

            if i >= 7:
                if contract_now.iloc[i - 7, 6] < contract_now.iloc[i, 6]:
                    Bank_money_bs = Bank_money_bs + (
                            contract_now.iloc[i - 7, 6] - contract_now.iloc[i, 6]) * state[0][2] * (1 + slip_point)
                else:
                    Bank_money_bs = Bank_money_bs + (
                            contract_now.iloc[i - 7, 6] - contract_now.iloc[i, 6]) * state[0][2] * (1 - slip_point)
                if delta_ddpg_list[-2] < delta_ddpg_list[-1]:
                    Bank_money_ddpg = Bank_money_ddpg + (delta_ddpg_list[-2] - delta_ddpg_list[-1]) * \
                                      state[0][2] * (1 + slip_point)
                else:
                    Bank_money_ddpg = Bank_money_ddpg + (delta_ddpg_list[-2] - delta_ddpg_list[-1]) * \
                                      state[0][2] * (1 - slip_point)
            Bank_money_bs = Bank_money_bs * bank_factor
            Bank_money_ddpg = Bank_money_ddpg * bank_factor
            error_ddpg = np.abs(V_next - action * S_next - Bank_money_ddpg)
            error_ddpg = error_ddpg[0][0]
            error_bs = np.abs(V_next - delta_BS * S_next - Bank_money_bs)
            state_next = [maturity_next, sigma_next, S_next, delta_ddpg_list[-1],
                          V_next, delta_BS_next, vega_next, K]
            state_next = np.reshape(state_next, (1, state_dim))
            print(delta_BS)
            if use_enderror == 1:
                if i == max_i - 7:
                    if 0.05 < np.abs(delta_BS) < 0.95:
                        # print('juju')
                        bucket_num = classification(delta_BS)
                        bucket.setdefault(bucket_num, []).append([error_ddpg, error_bs])
                        bucket.setdefault('all', []).append([error_ddpg, error_bs])
            else:
                if 0.05 < np.abs(delta_BS) < 0.95:
                    # print('juju')
                    bucket_num = classification(delta_BS)
                    bucket.setdefault(bucket_num, []).append([error_ddpg, error_bs])
                    bucket.setdefault('all', []).append([error_ddpg, error_bs])

            state = state_next
            delta_BS = delta_BS_next
            i += 7
            if i == max_i:
                break
    print(bucket.keys())
    # calculate Gain
    Gain = {}
    bucket_num = {}
    # print(bucket.keys())
    for p in range(1, 10):
        Gain[p] = 1 - np.sum([j[0] ** 2 for j in bucket[p]]) / np.sum([j[1] ** 2 for j in bucket[p]])
    Gain['all'] = 1 - np.sum([j[0] ** 2 for j in bucket['all']]) / np.sum([j[1] ** 2 for j in bucket['all']])
    for t in range(1, 10):
        bucket_num[t] = len(bucket[t])
    return Gain, bucket_num


def main(use_bs=is_bs, option_type=type_option):
    """
    Every time Sampling,the episode will stop once at the end of one contract or the score less than
    the specific lower limit.

    :return:
    """
    with tf.Session() as sess:
        # bucket_train = {}
        Gain = {}
        for p in range(1, 10):
            Gain[p] = 0
        Gain['all'] = 0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for n in range(1, train_num + 1):
            print('第{}轮训练。。。'.format(n))
            step = 0
            contract_flag = 0
            temp = 0
            done = 0
            i = 0
            score = 0
            while temp == contract_flag:
                temp = np.random.randint(0, int(2 / 3 * (len(contract) - 1)))
            contract_flag = foot[temp]
            contract_now = contract[contract_flag]
            max_i = contract_now.shape[0] - 1
            if max_i <= 14:
                continue
            ####过滤异常delta的合约######
            if option_type == 'C':
                if np.mean(contract_now.iloc[:, 6]) < 0.05 or np.mean(contract_now.iloc[:, 6]) > 0.95:
                    continue
            if option_type == 'P':
                if np.mean(contract_now.iloc[:, 6]) > -0.05 or np.mean(contract_now.iloc[:, 6]) < -0.95:
                    continue
            delta_ddpg_list = []

            S = contract_now.iloc[0, 4]  # Underlying Price
            V = contract_now.iloc[0, 5]  # Option Price
            K = contract_now.iloc[0, 3]
            delta_BS = contract_now.iloc[0, 6]
            num_of_stock = 0
            sigma = contract_now.iloc[0, 11]
            vega = contract_now.iloc[0, 9]
            maturity = contract_now.iloc[0, 12]
            state = [maturity, sigma, S, num_of_stock, V, delta_BS, vega, K]
            state = np.reshape(state, (1, state_dim))
            action = agent.select_action(state)  # add OU noise

            if use_bs == 1:
                action = delta_BS + action / fenmu
            if option_type == 'C':
                action = np.clip(action, 0, 1)
            else:
                if option_type == 'P':
                    action = np.clip(action, -1, 0)
            Bank_money_ddpg = V - action * S * (1 + slip_point)
            Bank_money_bs = V - delta_BS * S * (1 + slip_point)
            while not done:
                step += 1
                print('one episode continue。。。')
                print('现在得分是：{}'.format(score))
                # print('reward',reward)
                if i + 7 > max_i:
                    i = max_i - 7
                S_next = contract_now.iloc[i + 7, 4]
                V_next = contract_now.iloc[i + 7, 5]
                delta_BS_next = contract_now.iloc[i + 7, 6]
                sigma_next = contract_now.iloc[i + 7, 11]
                vega_next = contract_now.iloc[i + 7, 9]
                maturity_next = contract_now.iloc[i + 7, 12]
                action = agent.select_action(state)
                print('输出的DDPG头寸为{}'.format(action))
                print('BS公式头寸为{}'.format(delta_BS))
                if use_bs == 1:
                    action = delta_BS + action / fenmu
                if option_type == 'C':
                    action = np.clip(action, 0, 1)
                    action = action[0][0]
                else:
                    if option_type == 'P':
                        action = np.clip(action, -1, 0)
                        action = action[0][0]
                delta_ddpg_list.append(action)

                if i >= 7:
                    if contract_now.iloc[i - 7, 6] < contract_now.iloc[i, 6]:
                        Bank_money_bs = Bank_money_bs + (
                                contract_now.iloc[i - 7, 6] - contract_now.iloc[i, 6]) * state[0][2] * (1 + slip_point)
                    else:
                        Bank_money_bs = Bank_money_bs + (
                                contract_now.iloc[i - 7, 6] - contract_now.iloc[i, 6]) * state[0][2] * (1 - slip_point)
                    if delta_ddpg_list[-2] < delta_ddpg_list[-1]:
                        Bank_money_ddpg = Bank_money_ddpg + (delta_ddpg_list[-2] - delta_ddpg_list[-1]) * \
                                          state[0][2] * (1 + slip_point)
                    else:
                        Bank_money_ddpg = Bank_money_ddpg + (delta_ddpg_list[-2] - delta_ddpg_list[-1]) * \
                                          state[0][2] * (1 - slip_point)

                Bank_money_bs = Bank_money_bs * bank_factor
                Bank_money_ddpg = Bank_money_ddpg * bank_factor
                error_ddpg = V_next - action * S_next - Bank_money_ddpg
                error_ddpg = error_ddpg[0][0]
                error_bs = V_next - delta_BS * S_next - Bank_money_bs
                print('DDPG误差为{}'.format(error_ddpg))
                print('BS误差为{}'.format(error_bs))

                state_next = [maturity_next, sigma_next, S_next, delta_ddpg_list[-1],
                              V_next, delta_BS_next, vega_next, K]
                state_next = np.reshape(state_next, (1, state_dim))
                D_B = (state[0][3] - state_next[0][3]) * \
                      state[0][2] * (1 + slip_point)

                reward = -kappa / 2 * np.power(DISCOUNT_FACTOR, -step) * 1 / 356 * (
                        (state_next[0][3] - state[0][5]) * (state_next[0][2] - state[0][2])) ** 2 \
                         + np.power(DISCOUNT_FACTOR, -step) * (
                                 state_next[0][3] * state_next[0][2] - state[0][3] * state[0][2]
                                 + D_B)
                if i == max_i - 7:
                    reward = np.power(DISCOUNT_FACTOR, -step) * (
                                state_next[0][3] * state_next[0][2] - state[0][3] * state[0][2]
                                + D_B)
                print('现在回报是{}'.format(reward))
                agent.store_transition([state, action, reward, state_next, done])

                if len(replay_buffer.memory) > BATCH_SIZE:
                    agent.train_model()

                state = state_next
                delta_BS = delta_BS_next
                i += 7

                if i == max_i:
                    done = 1

            if n % 10000 == 0:
                saver.save(sess, './para_model/SPX_' + str(n) + '_model')
            if n % 10000 == 0:
                plt.figure(num=str(n))
                plt.plot(model.critic.loss_critic, color='g', linewidth=0.5)
                plt.savefig('./loss_fig/critic_' + str(n) + '.png')
        # Out of Sample
        test_data_flag = np.arange(int(2 * len(contract) / 3), len(contract))
        BS_error_test, ddpg_error_test, dic_K_S_bs_test, dic_K_S_ddpg_test, dic_T_bs_test, dic_T_ddpg_test = test_performance(
            agent, test_data_flag)
        avg_error_BS_test, avg_error_ddpg_test = plot_figure('2', BS_error_test, ddpg_error_test, 'test',
                                                             'out_sample', index_now, type_option)
        Gain_test = calculate_gain(agent, test_data_flag)[0]
        Gain_test = sorted(Gain_test.items(), key=lambda x: x[1], reverse=True)
        bucket_num_test = calculate_gain(agent, test_data_flag)[1]
        # In Sample
        train_data_flag = np.arange(0, int(2 * len(contract) / 3))
        BS_error_train, ddpg_error_train, dic_K_S_bs_train, dic_K_S_ddpg_train, dic_T_bs_train, dic_T_ddpg_train = test_performance(
            agent, train_data_flag)
        avg_error_BS_train, avg_error_ddpg_train = plot_figure('1', BS_error_train, ddpg_error_train, 'train',
                                                               'in_sample', index_now, type_option)
        Gain_train = calculate_gain(agent, train_data_flag)[0]
        Gain_train = sorted(Gain_train.items(), key=lambda x: x[1], reverse=True)
        bucket_num_train = calculate_gain(agent, train_data_flag)[1]

        print('Average Hedging error for BS model In Sample{}'.format(avg_error_BS_train))
        print('Average Hedging error for DDPG model In Sample{}'.format(avg_error_ddpg_train))
        print('K/S分类In Sample:')
        print(dic_K_S_bs_train)
        print(dic_K_S_ddpg_train)
        print('T分类In Sample:')
        print(dic_T_bs_train)
        print(dic_T_ddpg_train)

        print('Average Hedging error for BS model Out of Sample{}'.format(avg_error_BS_test))
        print('Average Hedging error for DDPG model Out of Sample{}'.format(avg_error_ddpg_test))
        print('K/S分类Out of Sample:')
        print(dic_K_S_bs_test)
        print(dic_K_S_ddpg_test)
        print('T分类Out of Sample:')
        print(dic_T_bs_test)
        print(dic_T_ddpg_test)

        print('Gain Performance in Sample:')
        print(Gain_train)
        print('Gain Bucket items\' num in Sample:')
        print(bucket_num_train)
        print('Gain Performance out of Sample:')
        print(Gain_test)
        print('Gain Bucket items\' num out of Sample:')
        print(bucket_num_test)

        ntfy_data = 'Training completed.'
        response = requests.post('https://notify.run/2tIWdaItl7cWhQpB', data=ntfy_data)

if __name__ == '__main__':
    main()
