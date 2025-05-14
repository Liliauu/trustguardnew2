import subprocess
import itertools
import json
import os
from datetime import datetime
import pandas as pd

def run_experiment(config):
    """运行单个实验配置"""
    base_cmd = "python main.py"
    
    # 构建命令行参数
    cmd_args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd_args.append(f"--{key}")
        else:
            cmd_args.append(f"--{key} {value}")
    
    full_cmd = f"{base_cmd} {' '.join(cmd_args)}"
    print(f"\n执行命令: {full_cmd}")
    
    try:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"命令执行失败，错误信息：\n{result.stderr}")
            return None
        
        parsed_results = parse_results(result.stdout)
        if not parsed_results:
            print("未能解析到任何结果指标")
            return None
            
        return parsed_results
    except Exception as e:
        print(f"实验运行出错: {str(e)}")
        return None

def parse_results(output):
    """从输出中解析实验结果"""
    results = {}
    metrics = ['mcc', 'auc', 'acc_balanced', 'precision', 'f1_micro', 'f1_macro']
    
    try:
        lines = output.split('\n')
        for line in lines:
            # 打印每行输出以便调试
            print(f"解析输出行: {line}")
            for metric in metrics:
                if f"{metric}:" in line.lower():
                    try:
                        value = float(line.split(':')[1].strip())
                        results[metric] = value
                        print(f"成功解析指标 {metric}: {value}")
                    except (ValueError, IndexError) as e:
                        print(f"解析指标 {metric} 时出错: {str(e)}")
    except Exception as e:
        print(f"结果解析出错: {str(e)}")
    
    return results

def save_results(all_results, results_dir, timestamp):
    """保存实验结果"""
    if not all_results:
        print("警告：没有可保存的实验结果")
        return
    
    try:
        # 将配置展平
        flat_results = []
        for result in all_results:
            flat_result = {}
            # 展平配置
            for key, value in result.get('config', {}).items():
                flat_result[f'config_{key}'] = value
            # 添加指标
            for key, value in result.items():
                if key != 'config':
                    flat_result[key] = value
            flat_results.append(flat_result)
        
        # 创建DataFrame
        results_df = pd.DataFrame(flat_results)
        
        # 保存结果
        csv_path = os.path.join(results_dir, f"ablation_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n实验结果已保存至: {csv_path}")
        
        # 生成结果汇总
        if not results_df.empty:
            print("\n实验结果汇总:")
            # 只对数值列进行统计
            numeric_cols = results_df.select_dtypes(include=['float64', 'int64']).columns
            print(results_df[numeric_cols].describe())
        else:
            print("警告：结果DataFrame为空")
            
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")

def main():
    # 实验配置
    base_config = {
        "epochs": 50,
        "learning-rate": 0.005,
        "weight-decay": 1e-5,
    }
    
    # SMOTE配置
    smote_configs = [
        {},  # 不使用SMOTE
        {
            "use-smote": True,
            "smote-strategy": "auto",
            "random-seed": 42
        },
        {
            "use-smote": True,
            "smote-strategy": "minority",
            "random-seed": 42
        }
    ]
    
    # Node2Vec配置
    node2vec_configs = [
        {},  # 不使用Node2Vec
        {
            "use-node2vec": True,
            "node2vec-p": 1.0,
            "node2vec-q": 1.0,
            "node2vec-dimensions": 64
        },
        {
            "use-node2vec": True,
            "node2vec-p": 2.0,
            "node2vec-q": 0.5,
            "node2vec-dimensions": 128
        }
    ]
    
    # 创建结果目录
    results_dir = "ablation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备存储结果
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 运行所有配置组合
    total_experiments = len(smote_configs) * len(node2vec_configs)
    current_experiment = 0
    
    for smote_config in smote_configs:
        for node2vec_config in node2vec_configs:
            current_experiment += 1
            # 合并配置
            current_config = {**base_config, **smote_config, **node2vec_config}
            
            print(f"\n{'='*50}")
            print(f"运行实验 {current_experiment}/{total_experiments}")
            print("当前配置:")
            print(json.dumps(current_config, indent=2))
            
            # 运行实验
            results = run_experiment(current_config)
            
            if results:
                # 记录配置和结果
                experiment_record = {
                    "config": current_config,
                    **results
                }
                all_results.append(experiment_record)
                print(f"实验 {current_experiment} 完成，成功记录结果")
            else:
                print(f"实验 {current_experiment} 未能产生有效结果")
    
    # 保存和显示结果
    save_results(all_results, results_dir, timestamp)

if __name__ == "__main__":
    main()