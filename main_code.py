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
import os
import csv
#12 2 point more and 1 point more，0.4 start negative; 13 1.5几，0.4 about
fenmu = 13#the factor to mutiply the outcome of ActorNetwork,then add delta_BS
BATCH_SIZE = 2 ** 12
BUFFER_SIZE = 1e4
DISCOUNT_FACTOR = 0.98#discount factor
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
TAU = 0.001#soft update rate
bank_ratio = 0.05 / 365#Bank risk_free ratio
train_num =1e6
max_buffer_num = 1e5
slip_point=0
is_bs=0#0 if use ddpg model output delta directly else 1
diction='no_new'
type_option='C'
index_now='SPX_0'
option='SPX_C'
"""
Note that C is replaced by P option，Need to change Actor activation function!!!!!!
"""
#############Note SP500 the order of the data columns#########
####get training data###################
base_path=os.path.dirname(os.getcwd())
path = os.path.join(base_path, 'ddpg_daibing/train_data/SPX_C_train_filterdelta.csv')
contract_data = pd.read_csv(path).groupby(['RooT', 'expiration', 'strike'])
########################################
contract = [df for item, df in contract_data ]
# Upset the training and testing sample to eliminate the correlation
random.shuffle(contract)
#
print('Number of contracts: {}'.format(len(contract)))
state_dim = 2
action_dim = 1
#Initialize Agent,memory Buffer,OU process
#########put option's delta in（-1，0）##########
model = Model(state_dim,
              action_dim,
              actor_learning_rate=ACTOR_LEARNING_RATE,
              critic_learning_rate=CRITIC_LEARNING_RATE,
              tau=TAU)
replay_buffer = Replay_Buffer(buffer_size=int(BUFFER_SIZE), batch_size=BATCH_SIZE)
exploration_noise = OU_Process(action_dim,type_option)
agent = Agent(model, replay_buffer, exploration_noise, discout_factor=DISCOUNT_FACTOR)


def test_performance(agent, test_data_flag,use_bs=is_bs,option_type=type_option):
    """
    this function aims to retest the performance of BS model and DDPG
    model on training data and testing data
    :param agent:
    :param test_data_flag:
    :return:
    """
    error_BS_list = []
    error_ddpg_list = []
    mem=[]
    for flag in test_data_flag:  # Loop on all testing sample
        print('New contract...')
        contract_now = contract[flag]
        i = 0
        max_i = contract_now.shape[0] - 1
        delta_ddpg_list = []
        S = contract_now.iloc[0, 4]  # Underlying Price
        V = contract_now.iloc[0, 6]  # Option Price
        delta_BS = contract_now.iloc[0, 5]
        state = [S, V]
        state = np.reshape(state, (1, 2))
        Bank_money_bs = V - delta_BS * S*(1+slip_point)
        action = agent.predict_action(state)
        if use_bs==1:

            action=delta_BS +action / fenmu

        Bank_money_ddpg = V - action * S*(1+slip_point)

        while True:

            S_next = contract_now.iloc[i + 1, 4]
            V_next = contract_now.iloc[i + 1, 6]
            delta_BS = contract_now.iloc[i, 5]
            if np.isnan(delta_BS):
                delta_BS = contract_now.iloc[i-1, 4]
            action = agent.predict_action(state)
            a_test=action
            print('The output DDPG position is{}'.format(action))
            print('BS formula position is{}'.format(delta_BS))
            if use_bs == 1:
                action = delta_BS + action / fenmu

            delta_ddpg_list.append(action)

            if i > 0:

                Bank_money_bs = Bank_money_bs + (
                        contract_now.iloc[i - 1, 5] - contract_now.iloc[i, 5]) * state[0][0]*(1+slip_point)
                Bank_money_ddpg = Bank_money_ddpg + (delta_ddpg_list[-2] - delta_ddpg_list[-1]) * \
                                  state[0][0]*(1+slip_point)
            Bank_money_bs = Bank_money_bs * (1 + bank_ratio)
            Bank_money_ddpg = Bank_money_ddpg * (1 + bank_ratio)
            error_ddpg = V_next - action * S_next - Bank_money_ddpg
            error_bs = V_next - delta_BS * S_next - Bank_money_bs
            print('DDPG error is{}'.format(error_ddpg))
            print('BS error is{}'.format(error_bs))
            mem.append([delta_BS,error_bs,action,error_ddpg,a_test])
            state_next = [S_next, V_next]
            state_next = np.reshape(state_next, (1, 2))
            state = state_next
            i += 1
            if i == max_i:
                break

            if np.isnan(error_bs):
                print('An error occurred')
                print(V_next)
                print(delta_BS)
                print(S_next)
                print(Bank_money_bs)
                break

        if not np.isnan(error_bs):
            error_BS_list.append(error_bs)
            error_ddpg_list.append(error_ddpg)
    #####save data for analysis########
    with open('/ddpg_daibing/error_bs.csv','w',newline='') as t:
        writer=csv.writer(t)
        writer.writerow(error_BS_list)
    with open('/ddpg_daibing/error_ddpg.csv','w',newline='') as t_:
        writer=csv.writer(t_)
        writer.writerow(error_ddpg_list)
    with open('/ddpg_daibing/mem.csv','w',newline='') as t__:
        writer=csv.writer(t__)
        writer.writerows(mem)
    return error_BS_list, error_ddpg_list


def plot_figure(fig_number, bs_list, ddpg_list,type,txt_name,index_name):
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
    max_1=np.max(bs_list)
    max_2=np.max(ddpg_list)
    min_1=np.min(bs_list)
    min_2=np.min(ddpg_list)
    max=np.max([max_1,max_2])
    min=np.min([min_1,min_2])
    fig_all_start=np.max([np.abs(max),np.abs(min)])
    print(max_1,max_2,min_1,min_2,fig_all_start)
    ####plot partial error figure in [-85,85]#####
    y_bs = []
    y_ddpg = []
    x_local = np.arange(-85, 85, 5)
    for tick in x_local:
        temp = [itemm_1 for itemm_1 in bs_list if np.abs(itemm_1 - tick) <= 2.5]
        temp_ = [itemm_2 for itemm_2 in ddpg_list if np.abs(itemm_2 - tick) <= 2.5]
        u=len(temp)
        v=len(temp_)
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
    plt.savefig('./performance_' + type +index_name+'.png')
    print(x_bs)
    print(x_ddpg)
    print(y_bs)
    print(y_ddpg)
    #######plot complete error figure#######
    y_bs_ = []
    y_ddpg_ = []
    stride=int(2*int(fig_all_start)/34)
    x_local_= np.arange(-int(fig_all_start),int(fig_all_start) , stride)
    for tick in x_local_:
        temp = [itemm_1 for itemm_1 in bs_list if np.abs(itemm_1 - tick) <= stride/2]
        temp_ = [itemm_2 for itemm_2 in ddpg_list if np.abs(itemm_2 - tick) <= stride/2]
        u = len(temp)
        v = len(temp_)
        v = model.actor.layer_process(tick, u, v, type)
        y_bs_.append(u)
        y_ddpg_.append(v)
    bar_width = stride/2-0.5
    x_bs_ = x_local_ - bar_width / 2
    x_ddpg_ = x_local_ + bar_width / 2
    plt.figure(num=fig_number+'100')
    plt.bar(x_bs_, y_bs_, bar_width, align="center", color="c", label="BS_model", alpha=0.5)
    plt.bar(x_ddpg_, y_ddpg_, bar_width, color="b", align="center", label="ddpg_model", alpha=0.5)
    plt.xlabel("Terminal hedging error($)")
    plt.ylabel("contract_num")
    plt.legend()
    plt.savefig('./performance_all_' + type+'_'+index_name + '.png')
    #######calculate contract num in each interval#########
    y_bs__= []
    y_ddpg__= []
    stride_1=(-2.5+int(fig_all_start))/4
    bu_1=[]
    wps=-int(fig_all_start)
    bu_1.extend([wps,wps+stride_1,wps+2*stride_1,wps+3*stride_1,-2.5])
    bu_2 = list(map(lambda x:-x,bu_1))
    bu_2.sort()
    for i in range(len(bu_1)-1):
        temp = [itemm_1 for itemm_1 in bs_list if bu_1[i]<itemm_1 <bu_1[i+1]]
        temp_ = [itemm_2 for itemm_2 in ddpg_list if bu_1[i]<itemm_2 <bu_1[i+1]]
        y_bs__.append(len(temp))
        y_ddpg__.append(len(temp_))
    temp = [itemm_1 for itemm_1 in bs_list if -2.5<itemm_1<2.5  ]
    temp_ = [itemm_2 for itemm_2 in ddpg_list if  -2.5<itemm_2<2.5]
    y_bs__.append(len(temp))
    y_ddpg__.append(len(temp_))
    for i in range(len(bu_2)-1):
        temp = [itemm_1 for itemm_1 in bs_list if bu_2[i]<itemm_1 <bu_2[i+1]]
        temp_ = [itemm_2 for itemm_2 in ddpg_list if bu_2[i]<itemm_2 <bu_2[i+1]]
        y_bs__.append(len(temp))
        y_ddpg__.append(len(temp_))
    file = open('/ddpg_daibing/good/'+option+'/'+diction+'/'+txt_name+'.txt', 'w')
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
    for jj in range(4):
        print('Interval[{},{}]'.format(bu_1[jj],bu_1[jj+1]))
        print('Number of BS：{}'.format(y_bs__[0]))
        print('Number of DDPG：{}'.format(y_ddpg__[0]))
        y_ddpg__.pop(0)
        y_bs__.pop(0)
    print('Interval[-2.5,2.5]')
    print('Number of BS{}:'.format(y_bs__[0]))
    print('Number of DDPG{}'.format(y_ddpg__[0]))
    y_ddpg__.pop(0)
    y_bs__.pop(0)
    for jjj in range(4):
        print('Interval[{},{}]'.format(bu_2[jjj], bu_2[jjj + 1]))
        print('Number of BS：{}'.format(y_bs__[0]))
        print('Number of DDPG：{}'.format(y_ddpg__[0]))
        y_ddpg__.pop(0)
        y_bs__.pop(0)


    return avg_bs_error, avg_ddpg_error


def classification(delta_bs):
    """
    classify daily hedging performance to different bucket based on BS delta value

    :param delta_bs:BS delta
    :return:the bucket index for which the delta belongs to
    """
    if type_option=='P':
        bucket = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
    else:
        bucket = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    bucket_test = [np.abs(delta_bs - item) for item in bucket]
    floo = np.argmin(bucket_test)
    floo = (floo + 1)

    return floo


def calculate_gain(agent, data_flag,use_bs=is_bs,option_type=type_option):
    """
    calculate Gain value for different bucket

    :param agent:
    :param data_flag:
    :return:
    """
    bucket = {}
    for flag in data_flag:
        print('New contract...')
        contract_now = contract[flag]
        i = 0
        max_i = contract_now.shape[0] - 1
        delta_ddpg_list = []
        S = contract_now.iloc[0, 4]  # Target price
        V = contract_now.iloc[0, 6]  # Option price
        delta_BS = contract_now.iloc[0, 5]
        state = [S, V]
        state = np.reshape(state, (1, 2))
        Bank_money_bs = V - delta_BS * S*(1+slip_point)
        action = agent.predict_action(state)

        if use_bs==1:

            action=delta_BS +action / fenmu

        Bank_money_ddpg = V - action * S*(1+slip_point)

        while True:

            S_next = contract_now.iloc[i + 1, 4]
            V_next = contract_now.iloc[i + 1, 6]
            delta_BS = contract_now.iloc[i, 5]
            action = agent.predict_action(state)
            if use_bs == 1:
                action = delta_BS + action / fenmu

            delta_ddpg_list.append(action)

            if i > 0:
                Bank_money_bs = Bank_money_bs + (
                        contract_now.iloc[i - 1, 5] - contract_now.iloc[i, 5]) * state[0][0] * (1 + slip_point)
                Bank_money_ddpg = Bank_money_ddpg + (delta_ddpg_list[-2] - delta_ddpg_list[-1]) * \
                                  state[0][0]*(1+slip_point)
            Bank_money_bs = Bank_money_bs * (1 + bank_ratio)
            Bank_money_ddpg = Bank_money_ddpg * (1 + bank_ratio)
            error_ddpg = np.abs(V_next - action * S_next - Bank_money_ddpg)
            error_bs = np.abs(V_next - delta_BS * S_next - Bank_money_bs)
            state_next = [S_next, V_next]
            state_next = np.reshape(state_next, (1, 2))

            print(delta_BS)
            if 0.05< np.abs(delta_BS) < 0.95:
                bucket_num = classification(delta_BS)
                bucket.setdefault(bucket_num, []).append([error_ddpg, error_bs])
                bucket.setdefault('all', []).append([error_ddpg, error_bs])

            state = state_next
            i += 1
            if i == max_i:
                break
    print(bucket.keys())
    # calculate Gain
    Gain = {}
    bucket_num={}
    # print(bucket.keys())
    for p in range(1, 10):
        Gain[p] = 1 - np.sum([j[0] ** 2 for j in bucket[p]]) / np.sum([j[1] ** 2 for j in bucket[p]])
    Gain['all'] = 1 - np.sum([j[0] ** 2 for j in bucket['all']]) / np.sum([j[1] ** 2 for j in bucket['all']])
    for t in range(1, 10):
        bucket_num[t]=len(bucket[t])
    return Gain,bucket_num


def main(use_bs=is_bs,option_type=type_option):
    """
    Every time Sampling,the episode will stop once at the end of one contract or the score less than
    the specific lower limit.

    :return:
    """
    with tf.Session() as sess:
       # bucket_train = {}
        Gain = {}
        for p in range(1, 10):
            Gain[p] =0
        Gain['all'] =0
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for n in range(1, int(train_num) + 1):
            print('Training round {}...'.format(n))
            step=0
            contract_flag = 0
            temp = 0
            done = 0
            i = 0
            while temp == contract_flag:
                temp = np.random.randint(0, int(2/3*(len(contract)-1)))
            contract_flag = temp
            contract_now = contract[contract_flag]
            max_i = contract_now.shape[0] - 1
            delta_ddpg_list = []

            S = contract_now.iloc[0, 4]  # Underlying Price
            V = contract_now.iloc[0, 6]  # Option Price
            delta_BS = contract_now.iloc[0, 5]
            state = [S, V]
            state = np.reshape(state, (1, 2))
            action = agent.select_action(state)#add OU noise

            if use_bs == 1:
                action = delta_BS + action / fenmu
            if option_type == 'C':
                action = np.clip(action, 0, 1)
            else:
                if option_type == 'P':
                    action = np.clip(action, -1, 0)
            Bank_money_ddpg = V - action * S*(1+slip_point)
            Bank_money_bs = V - delta_BS * S*(1+slip_point)
            error_ddpg_pre=0
            while not done:
                step+=1
                print('one episode continue...')
                S_next = contract_now.iloc[i + 1, 4]
                V_next = contract_now.iloc[i + 1, 6]
                delta_BS = contract_now.iloc[i, 5]
                action = agent.select_action(state)
                print('The output DDPG position is{}'.format(action))
                print('BS formula position is{}'.format(delta_BS))
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

                if i > 0:
                    Bank_money_bs = Bank_money_bs + (
                            contract_now.iloc[i - 1, 5] - contract_now.iloc[i, 5]) * state[0][0] * (1 + slip_point)
                    Bank_money_ddpg = Bank_money_ddpg + (delta_ddpg_list[-2] - delta_ddpg_list[-1]) * \
                                      state[0][0]*(1+slip_point)
                Bank_money_bs = Bank_money_bs * (1 + bank_ratio)
                Bank_money_ddpg = Bank_money_ddpg * (1 + bank_ratio)
                error_ddpg=V_next - action * S_next - Bank_money_ddpg
                error_bs=V_next - delta_BS * S_next - Bank_money_bs
                print('DDPG error is{}'.format(error_ddpg))
                print('BS error is{}'.format(error_bs))

                state_next = [S_next, V_next]
                state_next = np.reshape(state_next, (1, 2))

                reward=(np.abs(error_ddpg_pre)-np.abs(error_ddpg))/10#subordinate reward
                if np.abs(error_ddpg)>=2.5:
                    reward-=np.abs(error_ddpg)/1000
                error_ddpg_pre=error_ddpg
                if i == max_i-1:
                    if np.abs(error_ddpg)<2.5:
                        reward+=1000#main sparse reward
                agent.store_transition([state, action, reward, state_next, done])

                if len(replay_buffer.memory) > BATCH_SIZE:
                    agent.train_model()

                state = state_next
                i += 1

                if i == max_i:
                    done=1

            if n % 10000 == 0:
                saver.save(sess, './para_model/SPX_' + str(n) + '_model')
            if n % 10000 == 0:
                plt.figure(num=str(n))
                plt.plot(model.critic.loss_critic, color='g', linewidth=0.5)
                plt.savefig('./loss_fig/critic_' + str(n) + '.png')
        # Out of Sample test
        test_data_flag = np.arange(int(2 * len(contract) / 3), len(contract))
        BS_error_test, ddpg_error_test = test_performance(agent, test_data_flag)
        avg_error_BS_test, avg_error_ddpg_test = plot_figure('2', BS_error_test, ddpg_error_test,'test',
                                                             'out_sample',index_now)
        Gain_test = calculate_gain(agent, test_data_flag)[0]
        bucket_num_test=calculate_gain(agent, test_data_flag)[1]
        #In Sample test
        train_data_flag = np.arange(0, int(2 * len(contract) / 3))
        BS_error_train, ddpg_error_train = test_performance(agent, train_data_flag)
        avg_error_BS_train, avg_error_ddpg_train = plot_figure('1', BS_error_train, ddpg_error_train,'train','in_sample',index_now)
        Gain_train = calculate_gain(agent, train_data_flag)[0]
        bucket_num_train = calculate_gain(agent, train_data_flag)[1]

        print('Average Hedging error for BS model In Sample{}'.format(avg_error_BS_train))
        print('Average Hedging error for BS model Out of Sample{}'.format(avg_error_BS_test))
        print('Average Hedging error for DDPG model In Sample{}'.format(avg_error_ddpg_train))
        print('Average Hedging error for DDPG model Out of Sample{}'.format(avg_error_ddpg_test))

        print('Gain Performance in Sample:')
        print(Gain_train)
        print('Gain Bucket items\' num in Sample:')
        print(bucket_num_train)
        print('Gain Performance out of Sample:')
        print(Gain_test)
        print('Gain Bucket items\' num out of Sample:')
        print(bucket_num_test)

if __name__ == '__main__':
    main()
