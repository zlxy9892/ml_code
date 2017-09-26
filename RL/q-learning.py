# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time


# parameters
N_STATES = 6                    # 1维世界的宽度
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9                   # 贪婪度 greedy
ALPHA = 0.1                     # 学习率
GAMMA = 0.9                     # 奖励递减值
MAX_EPISODES = 13               # 最大回合数
FRESH_TIME = 0.3                # 移动间隔时间


def create_Q_table(n_states, actions):
    table = pd.DataFrame(
            data=np.zeros(shape=(n_states, len(actions))),
            columns=actions
            )
    #print(table)
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = np.argmax(state_actions)
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2: # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S  # reach the left wall
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES-1) + ['T'] # '-----T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_step = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        #print('\r                           ', end='')
        print('\n')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = create_Q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0   # 每个回合的初始位置
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)   # 选择行为
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S, A]   # 估算的(状态-行为)值
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()    # 实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R         # 实际的(状态-行为)值 (回合结束)
                is_terminated = True # terminate this episode
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)      # 更新q_table
            S = S_  # 移动到下一个状态
            step_counter += 1
            update_env(S, episode, step_counter)    # 环境更新
    return q_table
    

### ---------------------- main ---------------------- ###
q_table = rl()
print('\n\nQ-table:\n')
print(q_table)
