import os
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

def process_folder(result_folder_path, total_result_folder):
    all_dirs = os.listdir(result_folder_path)
    
    # 分类处理不同类型的文件夹
    categories = {
        'gset': [d for d in all_dirs if d.startswith('gset')],
        'BA': [d for d in all_dirs if 'BA' in d.upper()],
        'ER': [d for d in all_dirs if 'ER' in d.upper()],
        'PL': [d for d in all_dirs if 'PL' in d.upper()]
    }

    for category, dirs in categories.items():
        summary_data = {}
        
        for dir_name in dirs:
            dir_path = os.path.join(result_folder_path, dir_name)
            method_name = dir_name.split('_')[-1].upper()
            
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.txt'):
                    parts = file_name.split('_')
                    
                    if category == 'gset':
                        graph_id = parts[1]
                        if graph_id not in summary_data:
                            summary_data[graph_id] = {}
                        file_path = os.path.join(dir_path, file_name)
                        data = extract_data_from_file(file_path)
                        if 'QUBO' in method_name:
                            summary_data[graph_id][f'GUROBI'] = data['obj']
                            summary_data[graph_id][f'Gap'] = data['gap']
                            summary_data[graph_id][f'Bound'] = data['obj_bound']
                        else:
                            summary_data[graph_id][method_name] = data['obj']
                    
                    else:
                        if category == 'BA':
                            node_count = int(parts[2])
                            id_number = parts[3][2:]
                        elif category == 'ER':
                            node_count = int(parts[2])
                            id_number = parts[3][2:]
                        elif category == 'PL':
                            node_count = int(parts[1])
                            id_number = parts[2][2:]
                        
                        if id_number not in summary_data:
                            summary_data[id_number] = {}
                        if node_count not in summary_data[id_number]:
                            summary_data[id_number][node_count] = {}
                        
                        file_path = os.path.join(dir_path, file_name)
                        data = extract_data_from_file(file_path)
                        
                        if 'QUBO' in method_name:
                            summary_data[id_number][node_count][f'GUROBI'] = data['obj']
                            summary_data[id_number][node_count][f'Gap'] = data['gap']
                            summary_data[id_number][node_count][f'Bound'] = data['obj_bound']
                        else:
                            summary_data[id_number][node_count][method_name] = data['obj']
        
        # 为每个类型生成对应的结果文件夹和CSV文件
        output_folder = os.path.join(total_result_folder, f'{category}_results')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if category == 'gset':
            df = pd.DataFrame.from_dict(summary_data, orient='index')
            df.index.name = 'Graph'
            df = df.sort_index()
            df.to_csv(os.path.join(output_folder, 'gset_summary.csv'))
        else:
            for id_number, node_data in summary_data.items():
                df = pd.DataFrame.from_dict(node_data, orient='index')
                df.index.name = 'Nodes'
                df = df.sort_index()  # 对Nodes进行从小到大的排序
                df.to_csv(os.path.join(output_folder, f'{category}_ID_{id_number}_summary.csv'))

if __name__ == "__main__":
    result_folder_path = r'./result'  # 替换为实际路径
    total_result_folder = r'./output'  # 替换为要存放结果的路径
    
    if os.path.exists(total_result_folder):
        shutil.rmtree(total_result_folder)  # 如果存在旧的结果文件夹，先删除
    os.makedirs(total_result_folder)
    
    process_folder(result_folder_path, total_result_folder)
