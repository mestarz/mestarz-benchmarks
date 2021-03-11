import gym
import numpy as np
import time
from model import SarsaAgent

def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs) # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        # time.sleep(0.5)
        # env.render()
        if done:
            break
    return total_reward
    
env_list = ["FrozenLake-v0"]

for env_name in env_list:

    # 使用gym创建迷宫环境，设置is_slippery为False降低环境难度
    env = gym.make(env_name)

    # 创建一个agent实例，输入超参数
    agent = SarsaAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)

    agent.restore(env_name)

    #查看算法效果
    reward = 0
    for episode in range(5000):
        reward = reward + test_episode(env, agent)
    print(env_name + '环境得分 %.4f' % (reward/5000))

