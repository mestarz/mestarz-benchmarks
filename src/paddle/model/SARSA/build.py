import gym
import numpy as np
import time
from model import SarsaAgent

def run_episode(env, agent, render=False):
    total_steps = 0 # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset() # 重置环境, 重新开一局（即开始新的一个episode）
    action = agent.sample(obs) # 根据算法选择一个动作

    while True:
        next_obs, reward, done, _ = env.step(action) # 与环境进行一个交互
        next_action = agent.sample(next_obs) # 根据算法选择一个动作
        # 训练 Sarsa 算法
        agent.learn(obs, action, reward, next_obs, next_action, done)

        action = next_action
        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1 # 计算step数
        if render:
            env.render() #渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


env_list = [("FrozenLake-v0", 5000)] #（环境名，训练次数）

for env_name, nums in env_list:
    # 使用gym创建迷宫环境，设置is_slippery为False会降低环境难度
    env = gym.make(env_name)  # 0 left, 1 down, 2 right, 3 up
    #env = gym.make(env_name)
    # 创建一个agent实例，输入超参数
    agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)

    # 训练500个episode
    for episode in range(nums):
        ep_reward, ep_steps = run_episode(env, agent)
    agent.save(env_name)
    print(env_name + '环境模型构建成功')
