import matplotlib.pyplot as plt
import os
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
        plt.figure(figsize=(10, 6))

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