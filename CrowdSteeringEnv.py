import gymnasium as gym
import pygame
import numpy as np
import random
import math
import tool
import collections
import copy
import colorsys

from collections import deque

with_af = True

class AgentPathQueue:
    def __init__(self, agent_num, history_length):
        """
        初始化代理位置队列
        :param agent_num: 代理的数量
        :param history_length: 每个代理保存的历史位置数量
        """
        self.agent_num = agent_num
        self.history_length = history_length
        # 初始化代理位置队列，每个代理一个deque
        self.path_queue = [deque(maxlen=history_length) for _ in range(agent_num)]

    def add_position(self, agent_id, position):
        """
        为指定代理添加当前位置到其历史位置队列中
        :param agent_id: 代理的ID（索引）
        :param position: 代理的当前位置（可以是任何可哈希或可比较的对象）
        """
        if self.check_agent_id(agent_id):
            self.path_queue[agent_id].append(position)

    def get_positions(self, agent_id):
        """
        获取指定代理的历史位置队列
        :param agent_id: 代理的ID（索引）
        :return: 代理的历史位置队列（deque）
        """
        if self.check_agent_id(agent_id):
            return list(self.path_queue[agent_id])  # 返回列表副本，避免外部修改

    def clear_agent_path(self, agent_id):
        """
        清除指定代理的历史位置数据
        :param agent_id: 代理的ID（索引）
        """
        if self.check_agent_id(agent_id):
            # 重新初始化deque，保持maxlen属性
            self.path_queue[agent_id] = deque(maxlen=self.history_length)

    def check_agent_id(self, agent_id):
        if 0 <= agent_id < self.agent_num:
            return True
        else:
            raise IndexError("Agent ID out of range")

    def __str__(self):
        """
        返回所有代理历史位置的字符串表示
        """
        return "\n".join(f"Agent {i}: {list(q)}" for i, q in enumerate(self.path_queue))


class CrowdSteeringEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, map_data=[], af_data=0, agent_num=1):
        self.map_data = map_data
        self.af_data = af_data.tolist()
        self.window_size = 1000  # 暂时默认方形，此为边长
        self.rows = len(map_data)
        self.cols = len(map_data[0])
        self.cell_size = self.window_size / self.rows  # 基于既定的窗口大小计算每一个cell的尺寸
        self.radius = self.cell_size / 2
        self.agent_num = agent_num
        self.pos = [0, 0] * agent_num
        self.v = [0.01, 0] * agent_num
        self.prop_speed = [0] * agent_num
        self.target = [0, 0] * agent_num
        self.start = [0, 0] * agent_num
        self.last_pos = self.pos.copy()  # 上一帧与目标的距离
        self.total_step = [0] * agent_num
        self.path = AgentPathQueue(agent_num, 200)
        self.color = self.generate_colors(self.agent_num)

        self.stack_size = 1

        self.single_obs_size = 22 if with_af else 14
        self.observation = [collections.deque(maxlen=self.stack_size) for _ in range(agent_num)]

        self.action_space = 360
        self.observation_space = self.stack_size * self.single_obs_size

        self.delta_t = 0.01  # 一个时间步长对应的时间，用于统一计算位移与速度积分
        self.max_speed = 3.0  # 最大移动速度
        self.perception_range = 120  # 感知范围
        self.perception_samples = 15  # 隔多少度采样一个位置
        self.perception_vision_angle = 120  # 视角探测角度，范围（0，360]，以当前朝向顺时针逆时针各转一半均可探测到
        self.perception = [-1] * (
                    self.perception_vision_angle // self.perception_samples + 1) * agent_num  # 例如180度sample是10度，则会有19个距离检测射线，包含0和180

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def get_base_angle(self, j):
        return tool.calculate_angle(tool.normalize(self.v[j]), [1, 0]) - self.perception_vision_angle / 2

    def perceive_map(self):
        for j in range(self.agent_num):
            self.perception[j] = [self.perception_range] * (self.perception_vision_angle // self.perception_samples + 1)
            for i in range(0, len(self.perception[j])):
                base_angle = self.get_base_angle(j)
                angle = base_angle + i * self.perception_samples
                if angle < 0:
                    angle = angle + 360
                if angle > 360:
                    angle = angle - 360
                cur_x = math.floor(self.pos[j][0] / self.cell_size)
                cur_y = math.floor(self.pos[j][1] / self.cell_size)

                # 沿伸方向
                if angle < 90 or angle > 270:
                    x_dir = 1
                elif angle == 90 or angle == 270:
                    x_dir = 0
                else:
                    x_dir = -1
                if angle > 180:
                    y_dir = 1
                elif angle == 180 or angle == 0:
                    y_dir = 0
                else:
                    y_dir = -1

                # 在x方向上沿伸
                tar_x = cur_x if x_dir == -1 else cur_x + 1
                angle = math.radians(angle)
                while x_dir != 0:
                    if tar_x < 0 or tar_x >= self.cols:
                        break
                    dis = math.fabs((tar_x * self.cell_size - self.pos[j][0]) / math.cos(angle))
                    if dis - self.radius > self.perception_range:
                        break
                    tar_y = math.floor((dis * math.fabs(math.sin(angle)) * y_dir + self.pos[j][1]) / self.cell_size)
                    if tar_y < 0 or tar_y >= self.rows:
                        if dis < self.perception[j][i] or math.fabs(self.perception[j][i] + 1) < 1e-6:
                            self.perception[j][i] = dis
                        break
                    if x_dir < 0:
                        tar_x = tar_x - 1
                    if tar_x < 0:
                        if dis < self.perception[j][i] or math.fabs(self.perception[j][i] + 1) < 1e-6:
                            self.perception[j][i] = dis
                        break
                    if self.map_data[tar_y][tar_x] == 1:
                        if dis < self.perception[j][i] or math.fabs(self.perception[j][i] + 1) < 1e-6:
                            self.perception[j][i] = dis
                        break
                    if x_dir > 0:
                        tar_x = tar_x + 1
                # 在y方向上沿伸
                tar_y = cur_y if y_dir < 0 else cur_y + 1
                while y_dir != 0:
                    if tar_y < 0 or tar_y >= self.rows:
                        break
                    dis = math.fabs((tar_y * self.cell_size - self.pos[j][1]) / math.sin(angle))
                    if dis - self.radius > self.perception_range:
                        break
                    tar_x = math.floor((dis * math.fabs(math.cos(angle)) * x_dir + self.pos[j][0]) / self.cell_size)
                    if tar_x < 0 or tar_x >= self.cols:
                        if dis < self.perception[j][i] or math.fabs(self.perception[j][i] + 1) < 1e-6:
                            self.perception[j][i] = dis
                        break
                    if y_dir < 0:
                        tar_y = tar_y - 1
                    if tar_y < 0:
                        if dis < self.perception[j][i] or math.fabs(self.perception[j][i] + 1) < 1e-6:
                            self.perception[j][i] = dis
                        break
                    if self.map_data[tar_y][tar_x] == 1:
                        if dis < self.perception[j][i] or math.fabs(self.perception[j][i] + 1) < 1e-6:
                            self.perception[j][i] = dis
                        break
                    if y_dir > 0:
                        tar_y = tar_y + 1
        return

    def perceive_agents(self):
        for j in range(self.agent_num):
            for i in range(0, len(self.perception[j])):
                base_angle = self.get_base_angle(j)
                angle = base_angle + i * self.perception_samples
                if angle < 0:
                    angle = angle + 360
                if angle > 360:
                    angle = angle - 360
                for k in range(self.agent_num):
                    if k != j:
                        v = [x - y for x, y in zip(self.pos[k], self.pos[j])]
                        dis = tool.distance(v, [0, 0])
                        if dis < 1e-2:
                            continue
                        ang_thres = math.atan(self.radius / dis)
                        real_angle = tool.calculate_angle(v, [1, 0])
                        d_angle = real_angle - angle
                        if math.fabs(math.radians(d_angle)) < ang_thres and dis < self.perception_range:
                            per = tool.calculate_side_ab(self.radius, dis, math.fabs(d_angle))
                            if per < self.perception[j][i]:
                                self.perception[j][i] = per

    def update_obs(self):
        for i in range(self.agent_num):
            p_t = [(x - y + 1000) / 2000 for x, y in zip(self.target[i], self.pos[i])]
            obs_v = [(x + self.max_speed) / (self.max_speed * 2) for x in self.v[i]]
            prop_v = [self.prop_speed[i] / self.max_speed]
            obs_perception = [x / self.perception_range for x in self.perception[i]]
            if with_af:
                x = math.floor(self.pos[i][1] / self.cell_size)
                y = math.floor(self.pos[i][0] / self.cell_size)
                data = p_t + obs_v + [self.prop_speed[i]] + obs_perception + self.af_data[x][y]
            else:
                data = p_t + obs_v + [self.prop_speed[i]] + obs_perception
            self.observation[i].append(data)

    def _get_obs(self):
        ret = []
        for i in range(self.agent_num):
            val = []
            if len(self.observation[i]) < self.stack_size:
                val = val + [0] * (self.stack_size - len(self.observation[i])) * self.single_obs_size
            for j in self.observation[i]:
                val = val + list(j)
            ret.append(val)
        return ret

    def _get_info(self):
        return {
            "muhaha"
        }

    @staticmethod
    def distance(point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def check_pos(self, x, y):
        if 0 <= x < self.cols and 0 <= y < self.rows:
            return True
        else:
            return False

    # 随机生成一个可移动的位置
    def generate_random_position(self):
        #参数，用于使得生成的位置远离边界
        thres = 1
        while True:
            row = random.randint(0 + thres, self.rows - 1 - thres)
            col = random.randint(0 + thres, self.cols - 1 - thres)
            if self.map_data[row][col] == 0 and self.map_data[row + 1][col] == 0 and self.map_data[row - 1][col] == 0\
                    and self.map_data[row][col + 1] == 0 and self.map_data[row][col - 1] == 0:
                return [col * self.cell_size + self.cell_size // 2, row * self.cell_size + self.cell_size // 2]

    def retask(self, num):
        self.pos[num] = self.generate_random_position()
        self.start[num] = copy.copy(self.pos[num])
        self.target[num] = self.generate_random_position()
        self.v[num] = [0.01, 0]
        self.prop_speed[num] = self.max_speed * (0.5 + random.random() * 0.5)
        self.last_pos[num] = [-1, -1]
        self.total_step[num] = 0
        self.observation[num].clear()
        observation = self._get_obs()
        self.path.clear_agent_path(num)
        return observation

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.pos = [0] * self.agent_num
        self.target = [0] * self.agent_num
        self.v = [0.01, 0] * self.agent_num
        self.prop_speed = [0] * self.agent_num
        self.last_pos = [0] * self.agent_num
        self.total_step = [0] * self.agent_num
        for i in range(self.agent_num):
            self.retask(i)
        self.perceive_map()
        self.perceive_agents()
        observation = self._get_obs()
        info = self._get_info()
        return observation

    def step(self, action):
        # action是一个0-35的数相当于对360度进行了采样，现在要将其还原为二维向量
        for i in range(self.agent_num):
            a = [j * self.delta_t * self.prop_speed[i] for j in action[i]]

            # 摩擦力
            fc = [self.v[i][0] * (-0.1), self.v[i][1] * (-0.1)]

            self.v[i] = [x + y + z for x, y, z in zip(self.v[i], a, fc)]
            if tool.distance(self.v[i], [0, 0]) > self.prop_speed[i]:
                self.v[i] = tool.normalize(self.v[i])
                self.v[i] = [x * self.prop_speed[i] for x in self.v[i]]
            self.pos[i] = [x + y * self.delta_t for x, y in zip(self.pos[i], self.v[i])]
            self.total_step[i] = self.total_step[i] + 1
            self.path.add_position(i, self.pos[i])
        self.perceive_map()
        self.perceive_agents()
        self.update_obs()
        reward = self.cal_reward()
        observation = self._get_obs()
        info = self._get_info()

        terminated, arrived = self.check_terminated()
        self.last_pos = self.pos.copy()

        return observation, reward, terminated, arrived, info

    def check_collision(self, num):
        # 先判断agent之间的碰撞
        for i in range(self.agent_num):
            if i != num:
                if self.distance(self.pos[i], self.pos[num]) < self.radius * 1.2:
                    return True

        x_val = [-self.radius / 2, 0, self.radius / 2, 0]
        y_val = [0, -self.radius / 2, 0, self.radius / 2]
        pos_x = [self.pos[num][0] + x for x in x_val]
        pos_y = [self.pos[num][1] + x for x in y_val]
        for x, y in zip(pos_x, pos_y):
            col = math.floor(x / self.cell_size)
            row = math.floor(y / self.cell_size)
            if self.check_pos(col, row):
                if self.map_data[row][col] == 1:
                    return True
            else:
                return True
        return False

    def check_terminated(self):
        ret = []
        arrived = []
        for i in range(self.agent_num):
            distance = self.distance(self.pos[i], self.target[i])
            if distance < self.cell_size:
                arrived.append(True)
                ret.append(True)
                continue
            else:
                arrived.append(False)
            if self.check_collision(i):
                ret.append(True)
                continue
            if self.total_step[i] > 8000:
                ret.append(True)
            else:
                ret.append(False)
        return ret, arrived

    def cal_reward(self):
        reward = [0] * self.agent_num
        for i in range(self.agent_num):
            distance = self.distance(self.pos[i], self.target[i])
            if distance < self.cell_size:
                # 计算难度来控制reward大小
                dis = self.distance(self.start[i], self.target[i])
                reward[i] = 150 #dis * 888 / self.total_step[i]
                continue

            # 运动方向对准加分
            if self.last_pos[i][0] > 0:
                v = [x - y for x, y in zip(self.target[i], self.last_pos[i])]
                v2 = [x - y for x, y in zip(self.pos[i], self.last_pos[i])]
                speed = tool.distance(v, [0, 0])
                speed2 = tool.distance(v2, [0, 0])
                if speed > 1e-2 and speed2 > 1e-2:
                    angle = tool.calculate_angle(v, v2)
                    angle_cos = math.cos(math.radians(angle))
                    if angle_cos > 0:
                        reward[i] += angle_cos * 0.2

            # 距离变近加分
            if self.last_pos[i][0] > 0:
                v = [x - y for x, y in zip(self.target[i], self.last_pos[i])]
                v2 = [x - y for x, y in zip(self.target[i], self.pos[i])]
                dis_last = tool.distance(v, [0, 0])
                dis_cur = tool.distance(v2, [0, 0])
                if dis_cur < dis_last:
                    max_val = self.prop_speed[i] * self.delta_t
                    reward[i] += min(dis_last - dis_cur, max_val) / max_val * 0.5

            # 与合适的速度接近的奖励
            speed = tool.distance(self.v[i], [0, 0])
            speed_dev = math.fabs(speed - self.prop_speed[i])
            #if speed_dev < 0.5:
            #    reward[i] = reward[i] + (math.exp(-speed_dev) - math.exp(-0.5)) * 1.5
            reward[i] = reward[i] + (1 - math.exp(speed_dev * 0.5)) * 0.15

            # 行动惩罚
            reward[i] += -0.7
            # 改为不动就惩罚
            # reward[i] += -abs(self.prop_speed[i] - tool.distance(self.v[i], [0, 0])) * 0.4

            # 障碍过近惩罚
            # 从中间射线向两侧权值依次减轻
            rate = [0.0108, 0.0192, 0.0252, 0.0288, 0.03, 0.0288, 0.0252, 0.0192, 0.0108]
            index = 0
            for j in self.perception[i]:
                if self.radius < j < self.radius * 3:
                    reward[i] = reward[i] - math.exp((self.radius - j) / self.radius) * rate[index] * 8
                    index = index + 1

            # 检测碰撞
            if self.check_collision(i) or self.total_step[i] >= 6000:
            #if self.check_collision(i):
                reward[i] = -10

        return reward

    def render(self, screen):
        if self.render_mode == "rgb_array":
            return self._render_frame(screen)

    def draw_map(self, screen):
        screen.fill((255, 255, 255))
        for i in range(self.rows):
            for j in range(self.cols):
                color = (0, 0, 0) if self.map_data[i][j] == 1 else (255, 255, 255)
                pygame.draw.rect(screen, color, (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

    def _render_frame(self, screen):
        self.draw_map(screen)
        draw_perception = False
        draw_targetline = False
        draw_path = True
        draw_speed = False
        font = pygame.font.SysFont('Arial', 15)
        for i in range(self.agent_num):
            if draw_path:
                # 获取代理的路径
                path = self.path.get_positions(i)
                if path:
                    # 为当前代理选择一个颜色（循环使用颜色列表）
                    color_index = i % len(self.color)
                    color = self.color[color_index]

                    # 绘制路径中的线段
                    for j in range(len(path) - 1):
                        pygame.draw.line(screen, color, path[j], path[j + 1], 2)

            # 绘制当前速度
            if draw_speed:
                speed = tool.distance(self.v[i], [0, 0])
                #speed_text = font.render(str("{:.{}f}".format(speed, 1)), True, (255, 0, 0))
                speed_text = font.render(f"({speed:.1f}, {self.prop_speed[i]:.1f})", True, (255, 0, 0))
                speed_rect = speed_text.get_rect()
                speed_rect_pos = copy.copy(self.pos[i])
                speed_rect_pos[1] += 10
                speed_rect.center = (speed_rect_pos)
                screen.blit(speed_text, speed_rect)

            # debug用，绘制perception，从agent按照perception各个角度与距离长度画一条线，如果为-1或0则不绘制
            if draw_perception:
                for angle, dis in enumerate(self.perception[i]):
                    if dis != -1:
                        base_angle = self.get_base_angle(i)
                        angle = base_angle + angle * self.perception_samples
                        angle_rad = math.radians(angle)  # 将角度转换为弧度
                        end_point = [self.pos[i][0] + dis * math.cos(angle_rad),
                             self.pos[i][1] - dis * math.sin(angle_rad)]
                        pygame.draw.line(screen, (0, 255, 0), self.pos[i], end_point, 1)

            # debug用，绘制连接target的线条
            if draw_targetline:
                speed = tool.distance(self.v[i], [0, 0])
                color = 255 * speed / self.max_speed
                pygame.draw.line(screen, (255 - color, 0, color), self.pos[i], self.target[i], 1)

        for i in range(self.agent_num):
            pygame.draw.circle(screen, (255, 0, 255), self.pos[i], self.cell_size / 2)

        return

    def generate_colors(self, num_colors, saturation=1.0, value=1.0):
        """
        生成指定数量的区分度较高的颜色。

        参数:
        - num_colors: 要生成的颜色数量。
        - saturation: 颜色的饱和度（0.0到1.0之间）。
        - value: 颜色的亮度（0.0到1.0之间）。

        返回:
        - 一个包含(R, G, B)元组的列表，每个元组代表一种颜色。
        """
        colors = []
        hue_step = 360.0 / num_colors
        for i in range(num_colors):
            hue = i * hue_step
            rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
            colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
        return colors

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def reset_diametric(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.pos = [0] * self.agent_num
        self.target = [0] * self.agent_num
        self.v = [0.01, 0] * self.agent_num
        self.prop_speed = [0] * self.agent_num
        self.last_pos = [0] * self.agent_num
        self.total_step = [0] * self.agent_num
        for i in range(self.agent_num):
            self.retask(i)

        r = 200
        for i in range(0, self.agent_num):
            ang = i / self.agent_num * 2 * math.pi
            self.pos[i] = [500 + r * math.cos(ang), 500 + r * math.sin(ang)]
            self.target[i] = [500 + r * math.cos(ang + math.pi), 500 + r * math.sin(ang + math.pi)]

        self.perceive_map()
        self.perceive_agents()
        observation = self._get_obs()
        info = self._get_info()
        return observation

    def retask_hallway2(self, num):
        left = random.random() > 0.5
        self.pos[num] = [random.random() * 400, 400 + random.random() * 200]
        if not left:
            self.pos[num][0] = 1000 - self.pos[num][0]
        self.start[num] = copy.copy(self.pos[num])
        self.target[num] = [random.random() * 400, 400 + random.random() * 200]
        if left:
            self.target[num][0] = 1000 - self.target[num][0]
        self.v[num] = [0.01, 0]
        self.prop_speed[num] = self.max_speed * (0.3 + random.random() * 0.7)
        self.last_pos[num] = [-1, -1]
        self.total_step[num] = 0
        self.observation[num].clear()
        observation = self._get_obs()
        self.path.clear_agent_path(num)
        return observation

    def reset_hallway2(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.pos = [0] * self.agent_num
        self.target = [0] * self.agent_num
        self.v = [0.01, 0] * self.agent_num
        self.prop_speed = [0] * self.agent_num
        self.last_pos = [0] * self.agent_num
        self.total_step = [0] * self.agent_num
        for i in range(self.agent_num):
            self.retask_hallway2(i)
        self.perceive_map()
        self.perceive_agents()
        observation = self._get_obs()
        info = self._get_info()
        return observation

    def retask_hallway4(self, num):
        dir = random.randint(0, 3)
        left = random.randint(0, 1)
        x1 = random.random() * 400
        if left == 0:
            y1 = 400 + random.random() * 100
            y2 = 400 + random.random() * 100
        else:
            y1 = 500 + random.random() * 100
            y2 = 500 + random.random() * 100
        x2 = random.random() * 400

        if dir == 0:
            self.pos[num] = [x1, y1]
            self.target[num] = [1000 - x2, y2]
        elif dir == 1:
            self.pos[num] = [1000 - x1, y1]
            self.target[num] = [x2, y2]
        elif dir == 2:
            self.pos[num] = [y1, x1]
            self.target[num] = [y2, 1000 - x2]
        elif dir == 3:
            self.pos[num] = [y1, 1000 - x1]
            self.target[num] = [y2, x2]

        self.start[num] = copy.copy(self.pos[num])
        self.v[num] = [0.01, 0]
        self.prop_speed[num] = self.max_speed * (0.3 + random.random() * 0.7)
        self.last_pos[num] = [-1, -1]
        self.total_step[num] = 0
        self.observation[num].clear()
        observation = self._get_obs()
        self.path.clear_agent_path(num)
        return observation

    def reset_hallway4(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.pos = [0] * self.agent_num
        self.target = [0] * self.agent_num
        self.v = [0.01, 0] * self.agent_num
        self.prop_speed = [0] * self.agent_num
        self.last_pos = [0] * self.agent_num
        self.total_step = [0] * self.agent_num
        for i in range(self.agent_num):
            self.retask_hallway4(i)
        self.perceive_map()
        self.perceive_agents()
        observation = self._get_obs()
        info = self._get_info()
        return observation