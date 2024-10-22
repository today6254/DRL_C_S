import sys

import gym
import random
import pygame
import torch
#import pandas as pd
import os
import time
import struct
import numpy as np
import cProfile
import queue

from TD3 import TD3
import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from CrowdSteeringEnv import CrowdSteeringEnv

render = True
load_model = False
ModelIdex = 24000

action_size = 540
seed = 13

filename = []  # 文件名
af_filename = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("current device:" + device.__str__())


def make_train_data_filename():
    size = [0.05, 0.1, 0.15]
    date_str = "train_data\\2024-08-04\\"
    for i in size:
        for j in range(2):
            filename.append(date_str + str(i) + "_2\\map_" + str(j + 1) + ".txt")
            af_filename.append(date_str + str(i) + "_2\\map_" + str(j + 1) + ".dst")


def read_map_from_file(filename):
    map_data = []
    with open(filename, 'r') as file:
        for line in file:
            row = []
            for char in line.strip():
                if char.isdigit():
                    row.append(int(char))
            map_data.append(row)
    return map_data


def read_af_from_file(filename):
    with open(filename, "rb") as f:
        # 读取头部信息
        packed_data_head = f.read(8)
        rows, cols = struct.unpack('<ii', packed_data_head)

        # 读取数据
        packed_data = f.read()

    # 将打包的数据解包并转换为浮点数数组，注意这里的shape与af_data的shape相匹配
    af_data = np.frombuffer(packed_data, dtype=np.float32).reshape((rows, cols, 8)).copy(order='K')

    # 计算每个向量的范数
    norm_values = np.linalg.norm(af_data, axis=2)

    # 检查范数是否为零，如果为零则跳过该向量的计算
    mask = norm_values != 0

    # 只对范数不为零的向量进行归一化操作
    af_data[mask] = af_data[mask] / norm_values[mask, np.newaxis]

    return af_data


def main(agent_num=1):
    print('开始训练')

    expl_noise = 0.25
    save_interval = 2000  # interval to save model
    writer = SummaryWriter(log_dir='runs/exp')

    make_train_data_filename()
    # 初始化Pygame
    if render:
        pygame.init()
        screen = pygame.display.set_mode((1000, 1000))
        pygame.display.set_caption("人群仿真环境")

    episode = 0
    map_data = read_map_from_file(filename[0])
    af_data = read_af_from_file(af_filename[0])
    env = CrowdSteeringEnv("rgb_array", map_data, af_data, agent_num)
    state_dim = env.stack_size * env.single_obs_size
    action_dim = 2
    max_action = 1.0
    kwargs = {
        "env_with_Dead": True,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,
        "net_width": 1024,
        "a_lr": 8e-5,
        "c_lr": 8e-5,
        "Q_batchsize": 256,
    }
    print(kwargs)
    model = TD3(**kwargs)
    if load_model:
        model.load(ModelIdex)
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    for file_index in range(len(filename)):
        print("当前第" + str(file_index + 1) + "个地图")
        step = 0
        last_step = 0
        if file_index % 2 == 0:
            expl_noise = 0.25

        map_data = read_map_from_file(filename[file_index])
        af_data = read_af_from_file(af_filename[file_index])
        env = CrowdSteeringEnv("rgb_array", map_data, af_data, agent_num)
        state = env.reset()

        all_ep_r = []
        ep_r = []
        steps = [0] * agent_num

        s, done = env.reset(), False
        terminated = []

        arrive_history = queue.Queue(500)
        tot_dis_history = queue.Queue(1000)
        tot_step_history = queue.Queue(1000)
        tot_dis = 0
        tot_step = 0
        arrive_in500 = 0
        '''Interact & train'''
        while True:
            if episode / 3000 > file_index + 1:
                break
            if render:
                action = []
                for state in s:
                    state = np.array(state, dtype=np.float32)
                    a = model.select_action(state)
                    action.append(list(a))
                s_prime, r, terminated, arrived, info = env.step(action)
                for i in range(len(terminated)):
                    if terminated[i]:
                        s_prime = env.retask(i)
                env.render(screen)
                pygame.display.flip()
            else:
                for i in range(agent_num):
                    steps[i] += 1
                action = []
                for state in s:
                    state = np.array(state, dtype=np.float32)
                    a = (model.select_action(state) +
                         np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(-max_action, max_action)
                    action.append(list(a))
                s_prime, r, terminated, arrived, info = env.step(action)

                # Tricks
                for i in range(agent_num):
                    if r[i] <= -100:
                        r[i] = -1
                        replay_buffer.add(s[i], action[i], r[i], s_prime[i], True)
                    else:
                        replay_buffer.add(s[i], action[i], r[i], s_prime[i], False)

                if replay_buffer.size > 2000:
                    model.train(replay_buffer)

                if len(ep_r) < len(r):
                    ep_r = r
                else:
                    for i in range(agent_num):
                        ep_r[i] += r[i]

                flag = False
                count = 0
                for i in range(agent_num):
                    if terminated[i]:
                        # 计算平均损耗
                        if arrived[i]:
                            dis = env.distance(env.start[i], env.target[i])
                            tot_dis_history.put(dis)
                            tot_step_history.put(env.total_step[i])
                            tot_dis += dis
                            tot_step += env.total_step[i]

                        last_his = False
                        if arrive_history.full():
                            last_his = arrive_history.get()

                        # 同步退出队列
                        if last_his:
                            tot_dis -= tot_dis_history.get()
                            tot_step -= tot_step_history.get()

                        arrive_history.put(arrived[i])
                        if last_his != arrived[i]:
                            if last_his:
                                arrive_in500 -= 1
                            else:
                                arrive_in500 += 1
                        flag = True
                        count += 1
                        s_prime = env.retask(i)
                if flag:
                    for i in range(count):
                        episode += 1
                        expl_noise *= 0.999
                        '''plot & save'''
                        if (episode + 1) % save_interval == 0:
                            model.save(episode + 1)
                            # plt.plot(all_ep_r)
                            # plt.savefig('seed{}-ep{}.png'.format(random_seed,episode+1))最新进展
                            # plt.clf()

                    '''record & log'''
                    # all_ep_r.append(ep_r)
                    tmp_tot_dis = 0#防止除0
                    if tot_dis == 0:
                        tmp_tot_dis = 1
                    else:

                        tmp_tot_dis = tot_dis
                    for i in range(agent_num):
                        if terminated[i]:
                            rr = ep_r[i]
                            ep_r[i] = 0.
                            if len(all_ep_r) == 0:
                                all_ep_r.append(rr)
                            else:
                                all_ep_r.append(all_ep_r[-1] * 0.9 + rr * 0.1)
                            writer.add_scalar('s_ep_r', all_ep_r[-1], global_step=episode)
                            writer.add_scalar('ep_r', rr, global_step=episode)
                            writer.add_scalar('exploare', expl_noise, global_step=episode)
                            writer.add_scalar('arrive_in500', arrive_in500, global_step=episode)
                            writer.add_scalar('ave_cost', tot_step / tmp_tot_dis, global_step=episode)
                            print('seed:', seed, 'episode:', episode, 'score:', rr, 'step:', steps[i], 'max:',
                                  max(all_ep_r), 'arrive_in500:', arrive_in500,
                                  'ave_cost:', tot_step / tmp_tot_dis, 'succeed:', arrived[i])
                            steps[i] = 0

                #env.render(screen)
                #pygame.display.flip()

            s = s_prime
            #if arrive_in500 > 450:
            #    print("End.")
            #    sys.exit()


    env.close()

if __name__ == "__main__":
    main(50)
    #cProfile.run("main(1)")