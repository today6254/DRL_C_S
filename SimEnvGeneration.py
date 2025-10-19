import random
import numpy as np

import matplotlib.pyplot as plt

import random
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import os
from datetime import datetime

# Tkinterアプリケーションウィンドウを作成
window = tk.Tk()

# パラメータ
k = 10  # 生成するマップ数
n, m = 10, 10  # 各マップのサイズ
obs_ratio = 0.3  # 障害物の割合
obs_size = 1  # 障害物の最小サイズ：obs_size * obs_size
single_r = 1  # 単独障害物比率

# UIコントロールを作成
label_k = tk.Label(window, text="生成するマップ数 (k):")
label_k.grid(row=0, column=0)
entry_k = tk.Entry(window)
entry_k.grid(row=0, column=1)

label_n = tk.Label(window, text="マップサイズ (n):")
label_n.grid(row=1, column=0)
entry_n = tk.Entry(window)
entry_n.grid(row=1, column=1)

label_m = tk.Label(window, text="マップサイズ (m):")
label_m.grid(row=2, column=0)
entry_m = tk.Entry(window)
entry_m.grid(row=2, column=1)

label_obs_ratio = tk.Label(window, text="障害物比率 (obs_ratio):")
label_obs_ratio.grid(row=3, column=0)
entry_obs_ratio = tk.Entry(window)
entry_obs_ratio.grid(row=3, column=1)

label_obs_size = tk.Label(window, text="障害物最小サイズ (obs_size):")
label_obs_size.grid(row=4, column=0)
entry_obs_size = tk.Entry(window)
entry_obs_size.grid(row=4, column=1)

label_single_r = tk.Label(window, text="単独障害物比率 (single_r):")
label_single_r.grid(row=5, column=0)
entry_single_r = tk.Entry(window)
entry_single_r.grid(row=5, column=1)

# ボタンクリック時の処理を定義
def generate_maps():
    # 入力パラメータの取得
    k = int(entry_k.get())
    n = int(entry_n.get())
    m = int(entry_m.get())
    obs_ratio = float(entry_obs_ratio.get())
    obs_size = int(entry_obs_size.get())
    single_r = float(entry_single_r.get())

    if obs_size > 1:
        m = m // obs_size
        n = n // obs_size

    # 現在の作業ディレクトリをルートディレクトリパスとして取得
    root_directory_path = os.getcwd()

    # # ディレクトリ名を作成
    now = datetime.now()
    current_time = now.strftime("%H-%M")
    current_date = now.strftime("%Y-%m-%d")
    directory_name = f"{current_date}_{n}_{m}_{obs_ratio}_{obs_size}"
    directory_path = os.path.join(root_directory_path, directory_name)

    # ディレクトリが存在しなければ作成
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for i in range(k):
        map_data = generate_map(n, m, obs_ratio)
        if obs_size > 1:
            map_data = np.kron(map_data, np.ones((obs_size, obs_size), dtype=int))
        # ディレクトリを含むファイル名を構築
        filename = os.path.join(directory_path, f'map_{i + 1}.txt')
        save_map(map_data, filename)

        print(f"Map {i+1}:")
        visualize_map(map_data, 'white', 'black')
        print()

def check_connectivity(map_data):
    n = len(map_data)
    m = len(map_data[0])
    visited = [[False for _ in range(m)] for _ in range(n)]
    stack = []

    start_i, start_j = 0, 0
    for i in range(n):
        for j in range(m):
            if map_data[i][j] == 0:
                start_i, start_j = i, j
                break
        if map_data[start_i][start_j] == 0:
            break

    stack.append((start_i, start_j))
    visited[start_i][start_j] = True

    while stack:
        i, j = stack.pop()

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < n and 0 <= nj < m and not visited[ni][nj] and map_data[ni][nj] == 0:
                visited[ni][nj] = True
                stack.append((ni, nj))

    for i in range(n):
        for j in range(m):
            if map_data[i][j] == 0 and not visited[i][j]:
                return False
    return True

# 0 を通行可能領域、1 を障害物と定義。1 の割合は 60% を超えないことを推奨
def generate_map(n, m, ratio):
    map_data = [[1 for _ in range(m)] for _ in range(n)]
    total_cells = n * m

    num_zeros = int(total_cells * (1 - ratio + ratio * single_r))
    num_extra_ones = total_cells * ratio * single_r

    # 全体を 0 にするための開始点を設定
    map_data[0][0] = 0

    # 各位置が訪問済みかを記録
    visited = [[False for _ in range(m)] for _ in range(n)]
    visited[0][0] = True

    # 初期位置に隣接する位置を記録するスタック
    stack = [(random.randint(0, n), random.randint(0, m))]
    num_zeros -= 1

    # ランダムなフラッドフィル（拡張）アルゴリズムで 0 を埋める
    while len(stack) > 0:
        i, j = stack.pop()

        map_data[i][j] = 0

        if num_zeros <= 0:
            continue

        # 方向をランダムに並べ替える
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < n and 0 <= nj < m and not visited[ni][nj]:
                visited[ni][nj] = True
                stack.append((ni, nj))
                num_zeros -= 1

    for _ in range(int(num_extra_ones)):
        while True:
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            if map_data[i][j] == 0:
                map_data[i][j] = 1

                if check_connectivity(map_data):  # 連結性を確認
                    break
                else:
                    map_data[i][j] = 0
                    continue

    return map_data

def dfs(i, j, map_data, visited):
    n = len(map_data)
    m = len(map_data[0])

    if i < 0 or i >= n or j < 0 or j >= m or visited[i][j] or map_data[i][j] == 1:
        return

    visited[i][j] = True

    dfs(i-1, j, map_data, visited)
    dfs(i+1, j, map_data, visited)
    dfs(i, j-1, map_data, visited)
    dfs(i, j+1, map_data, visited)

def save_map(map_data, filename):
    with open(filename, 'w') as file:
        for row in map_data:
            file.write(' '.join(map(str, row)) + '\n')
    file.close()

def load_map(filename):
    map_data = []
    with open(filename, 'r') as file:
        for line in file:
            row = list(map(int, line.strip().split()))
            map_data.append(row)
    return map_data

def visualize_map(map_data, color_0, color_1):
    n = len(map_data)
    m = len(map_data[0])

    plt.figure(figsize=(m, m))
    plt.axis('off')

    cmap = plt.cm.colors.ListedColormap([color_0, color_1])

    full_map = np.ones((n, m))

    for i in range(n):
        for j in range(m):
            if map_data[i][j] == 0:
                full_map[i][j] = 0

    plt.imshow(full_map, cmap=cmap, aspect='equal')

    plt.show()

# マップ生成ボタンを作成
button_generate = tk.Button(window, text="マップを生成", command=generate_maps)
button_generate.grid(row=6, column=0, columnspan=2)

# メインループを開始
window.mainloop()
