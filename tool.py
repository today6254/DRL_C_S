import numpy as np
import math


def normalize(vec):
    dis = distance([0, 0], vec)
    if dis == 0:
        return vec
    vec = [vec[0] / dis, vec[1] / dis]
    return vec


def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def rotation_matrix_from_axis_angle(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s],
        [s, c]
    ])


def calculate_angle(vector1, vector2):
    # ベクトルの内積を計算
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # ベクトルの大きさ（ノルム）を計算
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    # 角度のラジアン値を計算
    val = dot_product / (magnitude1 * magnitude2)
    # val が [-1, 1] の範囲に収まるようにする
    val = max(-1, min(1, val))
    radians = math.acos(val)

    # 時計回りか反時計回りかを判定
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    if cross_product < 0:
        radians = 2 * math.pi - radians

    # ラジアンを度に変換
    degrees = math.degrees(radians)

    # 結果が 0〜359 の範囲になるようにする
    return degrees % 360

def calculate_side_ab(side_ac, side_bc, angle_b):
    if angle_b == 0:
        return side_bc - side_ac
    # 角度をラジアンに変換
    angle_b = math.radians(angle_b)

    # 正弦定理を使って辺 AB の長さを計算
    angle_a = math.pi - math.asin(side_bc * math.sin(angle_b) / side_ac)
    angle_c = math.pi - angle_b - angle_a
    side_ab = side_bc * math.sin(angle_c) / math.sin(angle_a)

    return side_ab