import numpy as np
from gcn import GCNTrainer
from arg_parser import parameter_parser
from utils import tab_printer, read_graph, best_printer, setup_features, process_incremental_data
import os


def main():
    """主函数，用于执行图卷积网络（GCN）的训练和评估流程。
    步骤包括：
    1. 解析参数。
    2. 打印参数表。
    3. 读取图数据。
    4. 设置特征。
    5. 初始化最佳性能指标列表。
    6. 根据单次预测或多时段预测设置训练轮次。
    7. 循环进行模型训练和评估，记录每轮的最佳性能。
    8. 打印每轮的最佳结果。"""
    args = parameter_parser()#解析参数
    tab_printer(args)#打印参数表
    
    # 设置默认值
    if not hasattr(args, 'memory_capacity'):
        args.memory_capacity = 1000
    if not hasattr(args, 'memory_weight'):
        args.memory_weight = 0.5
        
    # 打印记忆相关参数
    print("+---------------------+-------------------------------+")
    print("| Memory Capacity     | {:<29} |".format(args.memory_capacity))
    print("+---------------------+-------------------------------+")
    print("| Memory Weight       | {:<29} |".format(args.memory_weight))
    print("+---------------------+-------------------------------+")
    
    #读取图数据
    edges = read_graph(args)  # number of edges --> otc: 35592, alpha: 24186
    # 设置特征
    setup_features(args)

    # 初始化最佳性能指标列表，并添加表头。
    best = [["# Training Timeslots", "Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro", "Run Time"]]

    # 根据单次预测或多时段预测设置训练轮次。
    times = 8 if args.single_prediction else 6
    # single_prediction: {1-2}-->3, {1-3}-->4, {1-4}-->5, {1-5}-->6, {1-6}-->7, {1-7}-->8, {1-8}-->9, {1-9}-->10
    # multi_prediction: {1-2}-->{3-5}, {1-3}-->{4-6}, {1-4}-->{5-7}, {1-5}-->{6-8}, {1-6}-->{7-9}, {1-7}-->{8-10}
    for t in range(times):
        # 构建并训练 GCN模型。
        trainer = GCNTrainer(args, edges)
        # 构建并设置数据集。
        trainer.setup_dataset()

        # 开始模型训练和评估，并记录每轮的最佳性能。
        print("Ready, Go! Round = " + str(t))
        trainer.create_and_train_model()

        best_epoch = [0, 0, 0, 0, 0, 0, 0]#初始化最优结果
        for i in trainer.logs["performance"][1:]:
            # sum of MCC, AUC, ACC_Balanced, F1_Macro,选出最优epoch
            if float(i[1]+i[2]+i[3]+i[6]) > (best_epoch[1]+best_epoch[2]+best_epoch[3]+best_epoch[6]):
                best_epoch = i

        # 添加运行时间和训练时段
        best_epoch.append(trainer.logs["training_time"][-1][1])
        best_epoch.insert(0, t + 2)
        best.append(best_epoch)

        # 迭代到下一轮
        args.train_time_slots += 1

    # 打印每轮的最佳结果。
    print("\nBest results of each run")
    best_printer(best)

    # 计算平均值，最大值，最小值和标准差
    print("\nMean, Max, Min, Std")
    analyze = np.array(best)[1:, 1:].astype(np.float64)
    mean = np.mean(analyze, axis=0)
    maxi = np.amax(analyze, axis=0)
    mini = np.amin(analyze, axis=0)
    std = np.std(analyze, axis=0)
    results = [["Epoch", 'MCC', "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro", "Run Time"], mean, maxi, mini, std]

    best_printer(results)


def test_incremental_update():
    """测试增量更新功能"""
    args = parameter_parser()
    tab_printer(args)
    edges = read_graph(args)
    
    # 初始化性能指标列表
    incremental_results = [["# Update Times", "MCC", "AUC", "ACC_Balanced", "F1_Micro", "F1_Macro", "Update_Time"]]
    
    # 初始训练
    trainer = GCNTrainer(args, edges)
    trainer.setup_dataset()
    trainer.create_and_train_model()
    
    # 假设有新的增量数据路径
    new_edge_path = "TrustGuard/data/incremental_edges.txt"
    
    # 执行多次增量更新以获取统计信息
    num_updates = 5  # 执行5次增量更新
    for i in range(num_updates):
        # 执行增量更新
        results = trainer.create_and_train_model(new_edge_path=new_edge_path, is_incremental=True)
        
        # 将结果添加到列表中
        result_row = [i + 1, results["MCC"], results["AUC"], results["ACC_Balanced"], 
                     results["F1_Micro"], results["F1_Macro"], results["Update_Time"]]
        incremental_results.append(result_row)
    
    # 计算统计信息
    print("\n增量更新性能指标的统计分析:")
    analyze = np.array(incremental_results[1:]).astype(np.float64)
    mean = np.mean(analyze, axis=0)
    maxi = np.amax(analyze, axis=0)
    mini = np.amin(analyze, axis=0)
    std = np.std(analyze, axis=0)
    
    # 打印表头
    print("\n{:<15} {:<10} {:<10} {:<15} {:<12} {:<12} {:<12}".format(
        "Metric", "Mean", "Max", "Min", "Std", "Improvement", "Degradation"))
    print("-" * 85)
    
    # 打印每个指标的统计信息
    metrics = ["Update #", "MCC", "AUC", "ACC_Balanced", "F1_Micro", "F1_Macro", "Update_Time"]
    for idx, metric in enumerate(metrics):
        if idx == 0:  # 跳过Update #的统计
            continue
            
        # 计算相对于初始值的改善和退化
        initial_value = incremental_results[1][idx]
        improvement = ((mean[idx] - initial_value) / initial_value * 100) if initial_value != 0 else 0
        degradation = ((mini[idx] - initial_value) / initial_value * 100) if initial_value != 0 else 0
        
        print("{:<15} {:<10.4f} {:<10.4f} {:<15.4f} {:<12.4f} {:<12.2f}% {:<12.2f}%".format(
            metric, mean[idx], maxi[idx], mini[idx], std[idx], improvement, degradation))


if __name__ == "__main__":
    main()
    # 取消下面的注释以测试增量更新功能
    test_incremental_update()
