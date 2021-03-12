import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
from replay_memory import ReplayMemory
from model import Agent
from model import Model
from parl.algorithms import DQN


LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等
LEARNING_RATE = 0.001 # 学习率

# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


env_list = [('MountainCar-v0', 2000), ('CartPole-v0', 2000)] #（环境名，训练次数）

for env_name, max_episode in env_list:
    
    # 创建环境
	env = gym.make(env_name)
	action_dim = env.action_space.n  # MountainCar-v0: 3
	obs_shape = env.observation_space.shape  # MountainCar-v0: (2,)

	# 创建经验池
	rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

	# 根据parl框架构建agent
	model = Model(act_dim=action_dim)
	algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
	agent = Agent(
	    algorithm,
	    obs_dim=obs_shape[0],
	    act_dim=action_dim,
	    e_greed=0.1,
	    e_greed_decrement=1e-6
	)

	# 加载模型
	save_path = './DQN_checkpoint/'+env_name+'_dqn_model.ckpt'
	agent.restore(save_path)

	# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
	while len(rpm) < MEMORY_WARMUP_SIZE:
	    run_episode(env, agent, rpm)

	# 开始训练
	episode = 0
	while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
	    # train part
	    for i in range(0, 50):
	        total_reward = run_episode(env, agent, rpm)
	        episode += 1

	    # test part
	    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
	    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
	        episode, agent.e_greed, eval_reward))

	# 训练结束，保存模型
	save_path = './DQN_checkpoint/'+env_name+'_dqn_model.ckpt'
	agent.save(save_path)
	print(env_name + '环境模型构建成功')
