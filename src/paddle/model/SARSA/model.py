import gym
import numpy as np
import time

class SarsaAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n      # 动作维度，有几个动作可选
        self.lr = learning_rate # 学习率
        self.gamma = gamma      # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):

        if np.random.uniform() < (1.0 - self.epsilon):
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)
        
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):

        #action = max(self.Q[obs, :])
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]
        #np.where(Q_list == maxQ) 返回的是一个 1*n维的矩阵，n是满足条件的动作个数，所以要先通过[0]获取列表
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ on-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1
            done: episode是否结束
        """
        
        if done == True:
            #self.Q[obs, action] = reward
            target_Q = reward
        else:
            #self.Q[obs, action] += self.lr * (reward + self.gamma * self.Q[next_obs, next_action] - self.Q[obs, action])
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]
        
        self.Q[obs, action] += self.lr * (target_Q - self.Q[obs, action])

    # 保存Q表格数据到文件
    def save(self, name):
        npy_file = './SARSA_checkpoint/' + name + 'q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')
    
    # 从文件中读取数据到Q表格中
    def restore(self, name):
        npy_file='./SARSA_checkpoint/' + name + 'q_table.npy'
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')
