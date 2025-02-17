import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_scatter(logger_save_path):
    obj_values = []
    time_values = []
    time_step_values = []

    with open(logger_save_path, 'r') as f:
        for line in f:
            # 忽略注释行（以'//'开头的行）
            if line.startswith("//"):
                continue
            
            # 拆分每行数据并将其转换为浮动数
            obj, time_, time_step = map(float, line.split())
            
            # 将值添加到对应的列表
            obj_values.append(obj)
            time_values.append(time_)
            time_step_values.append(time_step)

        # 使用matplotlib绘图
        plt.figure(figsize=(16, 6))

        # 绘制obj随时间变化的图
        plt.subplot(2, 1, 1)
        plt.plot(time_values, obj_values, marker='o', color='b')
        plt.xlabel('Time')
        plt.ylabel('Obj')
        plt.title('Obj vs Time')

        # 绘制obj随time_step变化的图
        plt.subplot(2, 1, 2)
        plt.plot(time_step_values, obj_values, marker='o', color='r')
        plt.xlabel('Time Step')
        plt.ylabel('Obj')
        plt.title('Obj vs Time Step')

        plt.tight_layout()

        plot_save_path = os.path.splitext(logger_save_path)[0] + '.png'
        plot_save_path = plot_save_path.replace('result/eeco', 'result/eeco/plot')
        plot_dir = os.path.dirname(plot_save_path)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(plot_save_path, dpi=300)

def read_data_from_file(file_path):
    """从文件中读取数据并返回采样速度和时间"""
    times = []
    sampling_speeds = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过注释行
            if line.startswith("//"):
                continue
            # 解析数据：采样速度，时间，时间步
            parts = line.split()
            if len(parts) >= 3:
                sampling_speeds.append(float(parts[0]))  # 采样速度
                times.append(float(parts[2]))  # 时间
    
    return times, sampling_speeds

def plot_sampling_speed(folder_path):
    """绘制所有文件的采样速度随时间变化的图"""
    # 创建一个图表
    plt.figure(figsize=(16, 6))
    
    # 遍历文件夹中的所有 txt 文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            # 从文件中读取数据
            times, sampling_speeds = read_data_from_file(file_path)
            
            # 假设文件名包含环境数量，提取环境数量（从文件名中获取）
            n_sims = filename.split("_")[3]  # 假设文件名的第一个部分是环境数量
            
            # 在图表上绘制数据
            if n_sims == "1":
                plt.plot(times, sampling_speeds, label=f"cpu-env{n_sims}")
            else:    
                plt.plot(times, sampling_speeds, label=f"gpu-env{n_sims}")
    
    # 添加图表标题和标签
    plt.title('Sampling Speed vs Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Sampling Speed')
    
    # 显示图例
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 显示图表
    plt.grid(True)
    plt.savefig(folder_path + "/sampling_speed.png", dpi=300)

def plot_obj_vs_time(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 设置一个字典保存不同环境数量对应的数据
    data_dict = {}

    # 遍历每个文件
    for file in files:
        env_num = None
        with open(os.path.join(folder_path, file)) as f:
            lines = f.readlines()
            for line in lines:
                # 检查文件头部的n_sims值
                if line.startswith("//n_sims:"):
                    env_num = int(line.strip().split(":")[1])
                    break
            
            # 读取数据并保存到对应环境数量的字典中
            data = []
            for line in lines:
                if line.startswith("//"):
                    continue
                parts = line.split()
                obj = float(parts[0])
                time = float(parts[1])
                timestep = int(parts[2])
                data.append((time, timestep, obj))

            if env_num not in data_dict:
                data_dict[env_num] = []
            data_dict[env_num].extend(data)

    # 绘制图形
    plt.figure(figsize=(20, 6))

    for env_num, data in data_dict.items():
        df = pd.DataFrame(data, columns=["Time", "Timestep", "Obj"])
        if env_num == 1:
            plt.plot(df["Time"], df["Obj"], label=f'cpu-env(1)ECO')
        else:
            plt.plot(df["Time"], df["Obj"], label=f'gpu-env({env_num})EECO')

    plt.xlabel('Time')
    plt.ylabel('Obj')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('Obj vs Time for Different Environment Numbers')
    plt.grid(True)
    plt.savefig(folder_path + "/obj_vs_time.png", dpi=300)

def run():
       plot_obj_vs_time(r"RLSolver-master\rlsolver\result\eeco_obj_vs_time")
       plot_sampling_speed(r"RLSolver-master\rlsolver\result\eeco_sampling_speed")

if __name__ == "__main__":
    run()