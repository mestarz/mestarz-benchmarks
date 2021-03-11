import gym
import numpy as np
import time

class QLearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n      # 动作维度，有几个动作可选
        self.lr = learning_rate # 学习率
        self.gamma = gamma      # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        if np.random.rand() < (1 - self.epsilon):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        a_list = np.where(Q_list == max(Q_list))[0]
        action = np.random.choice(a_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            done: episode是否结束
        """
        current_q = self.Q[obs, action]
        if done == True:
            target = reward
        else:
            Q_list = self.Q[next_obs, :]
            target = reward + self.gamma * max(Q_list)
        self.Q[obs, action] = current_q + self.lr * (target - current_q)
    
    # 保存Q表格数据到文件
    def save(self, name):
        npy_file = './QLearning_checkpoint/' + name + 'q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')
    
    # 从文件中读取数据到Q表格中
    def restore(self, name):
        npy_file='./QLearning_checkpoint/' + name + 'q_table.npy'
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
