import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import shutil
import pandas as pd

def extract_data_from_file(file_path):
    data = {
        "obj": None,
        "gap": None,
        "obj_bound": None
    }

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("// obj:"):
                data["obj"] = float(line.split()[-1])
            elif line.startswith("// gap:"):
                data["gap"] = float(line.split()[-1])
            elif line.startswith("// obj_bound:"):
                data["obj_bound"] = float(line.split()[-1])
    return data

def process_folder(result_folder_path, total_result_folder, include_time=False, comparison_method=None, output_order=None):
    all_dirs = os.listdir(result_folder_path)

    categories={}
    for d in all_dirs:
        if 'BA' in d.upper():
            categories['BA']=[]
        elif 'ER' in d.upper():
            categories['ER']=[]
        elif 'PL' in d.upper():
            categories['PL']=[]
        else:
            categories[d.split('_')[0]]=[]
    for d in all_dirs:
        if 'BA' in d.upper():
            categories['BA'].append(d)
        elif 'ER' in d.upper():
            categories['ER'].append(d)
        elif 'PL' in d.upper():
            categories['PL'].append(d)
        else:
            categories[d.split('_')[0]].append(d)

    for category, dirs in categories.items():
        summary_data = {}

        for dir_name in dirs:
            dir_path = os.path.join(result_folder_path, dir_name)
            method_name = dir_name.split('_')[-1].upper()

            for file_name in os.listdir(dir_path):
                if file_name.endswith('.txt'):
                    parts = file_name.split('_')
                    time_taken = float(parts[-1].split('.')[0]) if include_time else None

                    if category not in ['BA', 'ER', 'PL']:
                        graph_id = parts[0]+'_'+parts[1]
                        if graph_id not in summary_data:
                            summary_data[graph_id] = {}
                        file_path = os.path.join(dir_path, file_name)
                        data = extract_data_from_file(file_path)
                        if 'QUBO' in method_name:
                            summary_data[graph_id][f'GUROBI'] = data['obj']
                            summary_data[graph_id][f'Gap'] = data['gap']
                            summary_data[graph_id][f'Bound'] = data['obj_bound']
                            if include_time:
                                summary_data[graph_id][f'GUROBI_Time'] = time_taken
                        else:
                            summary_data[graph_id][method_name] = data['obj']
                            if include_time:
                                summary_data[graph_id][f'{method_name}_Time'] = time_taken

                    else:
                        node_count = int(parts[2]) if category != 'PL' else int(parts[1])
                        id_number = int(parts[3][2:]) if category != 'PL' else int(parts[2][2:])

                        if node_count not in summary_data:
                            #这一条的作用是让顺序为0-30
                            # summary_data[node_count] = {}
                            summary_data[node_count] = {f'{i}': {} for i in range(3000)}


                        if f'{id_number}' not in summary_data[node_count]:
                            summary_data[node_count][f'{id_number}'] = {}

                        file_path = os.path.join(dir_path, file_name)
                        data = extract_data_from_file(file_path)

                        if 'QUBO' in method_name:
                            summary_data[node_count][f'{id_number}'][f'GUROBI'] = data['obj']
                            summary_data[node_count][f'{id_number}'][f'Gap'] = data['gap']
                            summary_data[node_count][f'{id_number}'][f'Bound'] = data['obj_bound']
                            if include_time:
                                summary_data[node_count][f'{id_number}'][f'GUROBI_Time'] = time_taken
                        else:
                            summary_data[node_count][f'{id_number}'][method_name] = data['obj']
                            if include_time:
                                summary_data[node_count][f'{id_number}'][f'{method_name}_Time'] = time_taken

        # 为每个类型生成对应的结果文件夹和CSV文件
        output_folder = os.path.join(total_result_folder, f'{category}_results')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if category not in ['BA', 'ER', 'PL']:
            df = pd.DataFrame.from_dict(summary_data, orient='index')
            df.index.name = 'Graph'
            df = df.sort_index()

            if comparison_method and comparison_method in df.columns:
                # 插入目标函数比较值行
                comparison_rows = []
                for graph_id, row in df.iterrows():
                    if not graph_id.endswith("_Comparison") and not graph_id.endswith("_Time_Comparison"):
                        comparison_values = {}
                        # 对obj列进行比较
                        for col in df.columns:
                            # 不包含时间列和Gap、Bound列
                            if  not col.endswith('_Time') and col not in ['Gap', 'Bound']:
                                if row[comparison_method] != 0 and not pd.isnull(row[col]):
                                    comparison_values[col] = (row[col] - row[comparison_method]) / row[comparison_method]
                                else:
                                    comparison_values[col] = None
                        comparison_df = pd.DataFrame(comparison_values, index=[f"{graph_id}_Comparison_{comparison_method}"])
                        comparison_rows.append(comparison_df)
                
                for comparison_df in comparison_rows:
                    df = pd.concat([df, comparison_df])

                # 插入时间比较值行（如果有GUROBI_Time和其他_Time列）
                time_comparison_rows = []
                if f'{comparison_method}_Time' in df.columns:
                    for graph_id, row in df.iterrows():
                        if not graph_id.endswith("_Comparison") and not graph_id.endswith("_Time_Comparison")and not graph_id.endswith(f'{comparison_method}'):
                            time_comparison_values = {}
                            comparison_time = row[f'{comparison_method}_Time']
                            if comparison_time and comparison_time != 0:
                                for col in df.columns:
                                    if col.endswith('_Time') :
                                        # if col == f'{comparison_method}_Time':
                                        #     print('skip')
                                        if not pd.isnull(row[col]):
                                            time_comparison_values[col] = (row[col] - comparison_time) / comparison_time
                                        else:
                                            time_comparison_values[col] = None
                            # 只有在有任何时间列比较值时才添加这一行
                            if time_comparison_values:
                                time_comparison_df = pd.DataFrame(time_comparison_values, index=[f"{graph_id}_Time_Comparison_{comparison_method}"])
                                time_comparison_rows.append(time_comparison_df)

                    for time_comparison_df in time_comparison_rows:
                        df = pd.concat([df, time_comparison_df])

            if output_order:
                ordered_columns = []
                for method in output_order:
                    if method == 'GUROBI':
                        gurobi_columns = [col for col in ["GUROBI", "Gap", "Bound", "GUROBI_Time"] if col in df.columns]
                        ordered_columns.extend(gurobi_columns)
                    elif method in df.columns:
                        ordered_columns.append(method)
                        if f"{method}_Time" in df.columns:
                            ordered_columns.append(f"{method}_Time")
                # 追加未在output_order中指定的列
                ordered_columns += [col for col in df.columns if col not in ordered_columns]
                df = df[ordered_columns]

            df.to_csv(os.path.join(output_folder, 'gset_summary.csv'))
        else:
            for node_count, id_data in summary_data.items():
                df = pd.DataFrame.from_dict(id_data, orient='index')
                df.index.name = 'ID'

                # df = df.sort_index()

                df.loc['Average'] = df.mean()  # 添加平均值行

                if include_time:
                    # 添加每种方法的平均时间行
                    time_columns = [col for col in df.columns if col.endswith('_Time')]
                    # for time_col in time_columns:
                    #     df.loc['Average_Time', time_col] = df[time_col].mean()

                    # 添加当前方法与对比方法的平均时间差
                    if comparison_method and f'{comparison_method}_Time' in df.columns:
                        comparison_avg_time = df[f'{comparison_method}_Time'].mean()
                        for time_col in time_columns:
                            if time_col != f'{comparison_method}_Time':
                                df.loc[f'Time_Comparison_{comparison_method}', time_col] = (df[time_col].mean() - comparison_avg_time) / comparison_avg_time

                if comparison_method and comparison_method in df.columns:
                    comparison_values = df.loc['Average']
                    diff_row = {}
                    for col in df.columns:
                        if col != comparison_method and not col.endswith('_Time') and col not in ['Gap', 'Bound']:
                            diff_row[col] = (comparison_values[col] - comparison_values[comparison_method]) / comparison_values[comparison_method]
                    df.loc[f'Obj_Comparison_{comparison_method}'] = pd.Series(diff_row)

                if output_order:
                    ordered_columns = []
                    for method in output_order:
                        if method == 'GUROBI':
                            gurobi_columns = [col for col in ["GUROBI", "Gap", "Bound", "GUROBI_Time"] if col in df.columns]
                            ordered_columns.extend(gurobi_columns)
                        elif method in df.columns:
                            ordered_columns.append(method)
                            if f"{method}_Time" in df.columns:
                                ordered_columns.append(f"{method}_Time")
                    ordered_columns += [col for col in df.columns if col not in ordered_columns]
                    df = df[ordered_columns]

                df.to_csv(os.path.join(output_folder, f'{category}_Nodes_{node_count}_summary.csv'))

if __name__ == "__main__":
    result_folder_path = r'../result'  # 替换为实际路径
    total_result_folder = r'./output'  # 替换为要存放结果的路径

    include_time = True  # 设置是否统计时间
    comparison_method = "GUROBI"  # 设置对比的方法名称
    output_order = ["GUROBI", "ECO-DQN","GA","GREEDY","S2V"]  # 设置表格列的输出顺序

    if os.path.exists(total_result_folder):
        shutil.rmtree(total_result_folder)  # 如果存在旧的结果文件夹，先删除
    os.makedirs(total_result_folder)

    process_folder(result_folder_path, total_result_folder, include_time=include_time, comparison_method=comparison_method, output_order=output_order)