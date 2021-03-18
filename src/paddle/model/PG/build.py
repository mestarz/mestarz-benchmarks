import os
import gym
import numpy as np
import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger
from parl.algorithms import PolicyGradient # 直接从parl库中导入PolicyGradient算法，无需重复写算法
from model import Model
from model import Agent
from model import LEARNING_RATE


def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs = preprocess(obs) # from shape (210, 160, 3) to (100800,)
        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = preprocess(obs) # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs) # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


# Pong 图片预处理
def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195] # 裁剪
    image = image[::2,::2,0] # 下采样，缩放2倍
    image[image == 144] = 0 # 擦除背景 (background type 1)
    image[image == 109] = 0 # 擦除背景 (background type 2)
    image[image != 0] = 1 # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float).ravel()


# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr

# 创建环境
env = gym.make('Pong-v0')
obs_dim = 80 * 80
act_dim = env.action_space.n
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

# 根据parl框架构建agent
model = Model(act_dim=act_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)


# 加载模型
if os.path.exists('./model.ckpt'):
    agent.restore('./model.ckpt')

for i in range(100):
    obs_list, action_list, reward_list = run_episode(env, agent)
    if i % 10 == 0:
        logger.info("Train Episode {}, Reward Sum {}.".format(i, 
                                             sum(reward_list)))

    batch_obs = np.array(obs_list)
    batch_action = np.array(action_list)
    batch_reward = calc_reward_to_go(reward_list)

    agent.learn(batch_obs, batch_action, batch_reward)
    if (i + 1) % 100 == 0:
        total_reward = evaluate(env, agent, render=False)
        logger.info('Episode {}, Test reward: {}'.format(i + 1, 
                                            total_reward))

# save the parameters to ./model.ckpt
agent.save('./model.ckpt')


# env_list = [('MountainCar-v0', 2000), ('CartPole-v0', 2000)] #（环境名，训练次数）

# for env_name, max_episode in env_list:
    
#     # 创建环境
# 	env = gym.make(env_name)
# 	action_dim = env.action_space.n  # MountainCar-v0: 3
# 	obs_shape = env.observation_space.shape  # MountainCar-v0: (2,)

# 	# 创建经验池
# 	rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 	# 根据parl框架构建agent
# 	model = Model(act_dim=action_dim)
# 	algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
# 	agent = Agent(
# 	    algorithm,
# 	    obs_dim=obs_shape[0],
# 	    act_dim=action_dim,
# 	    e_greed=0.1,
# 	    e_greed_decrement=1e-6
# 	)

# 	# 加载模型
# 	save_path = './DQN_checkpoint/'+env_name+'_dqn_model.ckpt'
# 	agent.restore(save_path)

# 	# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
# 	while len(rpm) < MEMORY_WARMUP_SIZE:
# 	    run_episode(env, agent, rpm)

# 	# 开始训练
# 	episode = 0
# 	while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
# 	    # train part
# 	    for i in range(0, 50):
# 	        total_reward = run_episode(env, agent, rpm)
# 	        episode += 1

# 	    # test part
# 	    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
# 	    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
# 	        episode, agent.e_greed, eval_reward))

# 	# 训练结束，保存模型
# 	save_path = './DQN_checkpoint/'+env_name+'_dqn_model.ckpt'
# 	agent.save(save_path)
# 	print(env_name + '环境模型构建成功')
