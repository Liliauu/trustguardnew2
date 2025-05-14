import torch
import numpy as np
from texttable import Texttable
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
import os
from imblearn.over_sampling import SMOTE


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def read_graph(args):
    """
    读取图数据并进行综合预处理，包括：
    1. 数据清洗：处理缺失值
    2. 异常值处理：检查节点ID和评分值
    3. 重复边处理：避免重复边
    4. 可选：使用SMOTE处理不平衡数据
    
    :param args: 参数对象，包含edge_path等参数
    :return: 处理后的边数据字典
    """
    edges = {}
    ecount = 0
    ncount = []
    edg = []
    lab = []
    unique_edges = set()  # 用于检测重复边
    
    print("开始数据预处理...")
    print(f"正在读取文件: {args.edge_path}")
    
    with open(args.edge_path) as dataset:
        for line_num, edge in enumerate(dataset, 1):
            parts = edge.split()
            
            # 1. 数据清洗：处理缺失值
            if len(parts) < 4:
                print(f"警告: 第{line_num}行数据格式不正确，已跳过")
                continue
                
            try:
                # 2. 异常值处理：检查节点ID和评分值
                source = int(parts[0])
                target = int(parts[1])
                rating_1 = int(parts[2])
                rating_2 = int(parts[3])
                
                # 检查节点ID是否有效
                if source < 0 or target < 0:
                    print(f"警告: 第{line_num}行包含无效的节点ID，已跳过")
                    continue
                    
                # 检查评分值是否有效
                if rating_1 not in [0, 1] or rating_2 not in [0, 1]:
                    print(f"警告: 第{line_num}行包含无效的评分值，已跳过")
                    continue
                
                # 3. 重复边处理
                edge_key = (source, target, rating_1, rating_2)
                if edge_key in unique_edges:
                    print(f"警告: 第{line_num}行是重复边，已跳过")
                    continue
                unique_edges.add(edge_key)
                
                # 添加有效边
                ecount += 1
                ncount.append(str(source))
                ncount.append(str(target))
                edg.append([source, target])
                lab.append([rating_1, rating_2])
                
            except ValueError:
                print(f"警告: 第{line_num}行数据无法转换为数值，已跳过")
                continue
    
    # 转换为NumPy数组
    edges["labels"] = np.array(lab)
    edges["edges"] = np.array(edg)
    edges["ecount"] = ecount
    edges["ncount"] = len(set(ncount))
    
    # 4. 可选：使用SMOTE处理不平衡数据
    if hasattr(args, 'use_smote') and args.use_smote:
        print("检测到不平衡数据，应用SMOTE技术...")
        sampling_strategy = getattr(args, 'smote_strategy', 'auto')
        random_state = getattr(args, 'random_seed', 42)
        
        # 应用SMOTE
        balanced_edges, balanced_labels = apply_smote(
            edges["edges"], 
            edges["labels"], 
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        
        # 更新边数据
        edges["edges"] = balanced_edges
        edges["labels"] = balanced_labels
        edges["ecount"] = len(balanced_edges)
        
        print(f"SMOTE处理后: {edges['ecount']} 条边, {edges['ncount']} 个节点")
    
    return edges

def setup_features(args):
    """
    Setting up the node features as a numpy array.
    :param args: Arguments object.
    :return X: Node features.
    """
    # 检查是否需要使用Node2Vec嵌入
    if hasattr(args, 'use_node2vec') and args.use_node2vec:
        print("使用Node2Vec生成节点嵌入...")
        return generate_node2vec_embeddings(args)
    
    # 默认使用随机嵌入
    print("使用随机初始化节点嵌入...")
    # otc: (5881,64), alpha: (3783,64)
    np.random.seed(args.seed)
    if args.data_path == "../data/bitcoinotc.csv":
        embedding = np.random.normal(0, 1, (5881, 64)).astype(np.float32)
    else:
        embedding = np.random.normal(0, 1, (3783, 64)).astype(np.float32)
    return embedding


def generate_node2vec_embeddings(args):
    """
    使用Node2Vec生成节点嵌入，包含缓存机制和优化的参数控制
    
    :param args: 参数对象，包含edge_path等参数
    :return: 节点嵌入向量
    """
    import torch
    import networkx as nx
    from node2vec import Node2Vec
    import pandas as pd
    import os
    from pathlib import Path
    import hashlib
    import json
    
    # 创建缓存目录
    cache_dir = Path("../cache/node2vec")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成缓存文件名（基于参数的哈希值）
    params_dict = {
        'p': args.node2vec_p,
        'q': args.node2vec_q,
        'dimensions': args.node2vec_dimensions,
        'walk_length': args.node2vec_walk_length,
        'num_walks': args.node2vec_num_walks,
        'window': args.node2vec_window,
        'edge_path': args.edge_path
    }
    params_str = json.dumps(params_dict, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    cache_file = cache_dir / f"embeddings_{params_hash}.npy"
    
    # 检查缓存
    if cache_file.exists():
        print(f"从缓存加载Node2Vec嵌入向量: {cache_file}")
        return np.load(cache_file)
    
    print(f"正在读取边数据: {args.edge_path}")
    
    # 从边数据创建图
    G = nx.Graph()
    edge_weights = {}
    
    # 读取边数据并构建加权图
    with open(args.edge_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                source, target = int(parts[0]), int(parts[1])
                # 使用信任值作为边权重（如果可用）
                weight = float(parts[2]) if len(parts) > 2 else 1.0
                
                # 更新边权重
                edge_key = tuple(sorted([source, target]))
                if edge_key in edge_weights:
                    edge_weights[edge_key] += weight
                else:
                    edge_weights[edge_key] = weight
    
    # 添加加权边到图中
    for (source, target), weight in edge_weights.items():
        G.add_edge(source, target, weight=weight)
    
    print(f"创建的图包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
    
    # 优化Node2Vec参数
    dimensions = args.node2vec_dimensions
    walk_length = args.node2vec_walk_length
    num_walks = args.node2vec_num_walks
    p = args.node2vec_p
    q = args.node2vec_q
    window = args.node2vec_window
    workers = args.node2vec_workers
    
    print(f"Node2Vec参数配置:")
    print(f"- 维度: {dimensions}")
    print(f"- 游走长度: {walk_length}")
    print(f"- 游走次数: {num_walks}")
    print(f"- 返回参数p: {p}")
    print(f"- 进出参数q: {q}")
    print(f"- 窗口大小: {window}")
    print(f"- 并行线程: {workers}")
    
    try:
        # 初始化Node2Vec模型
        node2vec = Node2Vec(
            graph=G,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers,
            quiet=False  # 显示训练进度
        )
        
        # 训练模型
        print("开始训练Node2Vec模型...")
        model = node2vec.fit(
            window=window,
            min_count=1,
            batch_words=4
        )
        
        # 确定节点数量
        if args.data_path == "../data/bitcoinotc.csv":
            num_nodes = 5881
        else:
            num_nodes = 3783
        
        # 初始化嵌入矩阵
        embeddings = np.zeros((num_nodes, dimensions), dtype=np.float32)
        
        # 填充嵌入矩阵，使用改进的初始化策略
        missing_nodes = []
        for node in range(num_nodes):
            if str(node) in model.wv:
                embeddings[node] = model.wv[str(node)]
            else:
                missing_nodes.append(node)
        
        if missing_nodes:
            print(f"警告: {len(missing_nodes)} 个节点没有嵌入向量")
            # 使用邻居平均值初始化缺失节点
            for node in missing_nodes:
                neighbors = list(G.neighbors(node))
                if neighbors:
                    neighbor_embeddings = [embeddings[n] for n in neighbors if n not in missing_nodes]
                    if neighbor_embeddings:
                        embeddings[node] = np.mean(neighbor_embeddings, axis=0)
                    else:
                        embeddings[node] = np.random.normal(0, 0.1, dimensions).astype(np.float32)
                else:
                    embeddings[node] = np.random.normal(0, 0.1, dimensions).astype(np.float32)
        
        # 保存到缓存
        np.save(cache_file, embeddings)
        print(f"Node2Vec嵌入向量已缓存至: {cache_file}")
        
        return embeddings
        
    except Exception as e:
        print(f"Node2Vec训练过程中出错: {str(e)}")
        print("回退到随机初始化...")
        return np.random.normal(0, 1, (num_nodes, dimensions)).astype(np.float32)


def calculate_auc(scores, label):
    label_vector = [i for line in label for i in range(len(line)) if line[i] == 1]
    prediction_vector = torch.argmax(scores, dim=1)

    acc_balanced = balanced_accuracy_score(label_vector, prediction_vector.cpu())
    mcc = matthews_corrcoef(label_vector, prediction_vector.cpu())

    f1_micro = f1_score(label_vector, prediction_vector.cpu(), average="micro")
    f1_macro = f1_score(label_vector, prediction_vector.cpu(), average="macro")

    prediction_pr = scores[:,1]
    auc = roc_auc_score(label_vector, prediction_pr.cpu().detach().numpy())
    precision = average_precision_score(label_vector, prediction_pr.cpu().detach().numpy())  # average precision
    return mcc, auc, acc_balanced, precision, f1_micro, f1_macro


def best_printer(log):
    t = Texttable()
    t.set_precision(4)
    t.add_rows([per for per in log])
    print(t.draw())


def process_incremental_data(args, new_edge_path):
    """
    处理增量边数据。
    
    :param args: 参数对象
    :param new_edge_path: 新边数据的文件路径
    :return: 处理后的新边数据和标签
    """
    new_edges = []
    new_labels = []
    
    # 检查文件是否存在
    if not os.path.exists(new_edge_path):
        print(f"警告: 增量数据文件 {new_edge_path} 不存在!")
        # 创建一个示例文件用于测试
        create_sample_incremental_file(new_edge_path)
        print(f"已创建示例增量数据文件: {new_edge_path}")
    
    try:
        with open(new_edge_path) as dataset:
            for line in dataset:
                parts = line.strip().split()
                if len(parts) >= 3:  # 至少需要源节点、目标节点和一个标签
                    new_edges.append(list(map(int, parts[0:2])))
                    new_labels.append(list(map(float, parts[2:])))
    except Exception as e:
        print(f"处理增量数据时出错: {str(e)}")
        return np.array([]), np.array([])
            
    if len(new_edges) == 0:
        print(f"警告: 从 {new_edge_path} 未加载到任何有效数据")
        return np.array([]), np.array([])
        
    print(f"从 {new_edge_path} 加载了 {len(new_edges)} 条新边")
    return np.array(new_edges), np.array(new_labels)

def create_sample_incremental_file(file_path):
    """
    创建一个示例增量数据文件用于测试
    
    :param file_path: 文件路径
    """
    # 确保目录存在
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 创建一些示例数据
    with open(file_path, 'w') as f:
        # 格式: 源节点ID 目标节点ID 标签1 [标签2 ...]
        f.write("0 1 1 0\n")
        f.write("1 2 0 1\n")
        f.write("2 3 1 0\n")
        f.write("3 0 0 1\n")
        f.write("4 5 1 0\n")

def precision_recall_curve(predictions, labels, n_points=100):
    """
    计算精确率-召回率曲线，找到最佳阈值
    
    :param predictions: 模型预测的概率
    :param labels: 真实标签
    :param n_points: 曲线上的点数
    :return: 最佳阈值和对应的F1分数
    """
    thresholds = np.linspace(0.1, 0.9, n_points)
    f1_scores = []
    precisions = []
    recalls = []
    
    # 获取正类的概率
    if isinstance(predictions, torch.Tensor):
        pos_probs = predictions[:, 1].detach().cpu().numpy()
    else:
        pos_probs = predictions[:, 1]
    
    # 处理标签，确保是一维数组
    if isinstance(labels, torch.Tensor):
        # 如果标签是二维的，取第二列（假设第二列是正类）
        if labels.dim() > 1:
            true_labels = labels[:, 1].detach().cpu().numpy()
        else:
            true_labels = labels.detach().cpu().numpy()
    else:
        # 如果是numpy数组，同样确保是一维的
        if len(labels.shape) > 1:
            true_labels = labels[:, 1]
        else:
            true_labels = labels
    
    print(f"预测概率形状: {pos_probs.shape}, 标签形状: {true_labels.shape}")  # 调试信息
    
    # 计算不同阈值下的精确率、召回率和F1
    for threshold in thresholds:
        pred_labels = (pos_probs >= threshold).astype(int)
        
        # 确保预测标签和真实标签都是一维的
        pred_labels = pred_labels.flatten()
        true_labels = true_labels.flatten()
        
        # 计算精确率和召回率
        true_positives = np.sum((pred_labels == 1) & (true_labels == 1))
        predicted_positives = np.sum(pred_labels == 1)
        actual_positives = np.sum(true_labels == 1)
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # 找到F1最高的阈值
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    
    print(f"最佳阈值: {best_threshold:.2f}, 精确率: {best_precision:.4f}, F1: {best_f1:.4f}")
    return best_threshold, best_f1

def apply_smote(edges, labels, sampling_strategy='auto', random_state=42):
    """
    使用SMOTE技术处理不平衡数据
    
    :param edges: 边数据，形状为(n_edges, 2)
    :param labels: 标签数据，形状为(n_edges,)
    :param sampling_strategy: 采样策略，默认为'auto'
    :param random_state: 随机种子
    :return: 平衡后的边数据和标签
    """
    print("应用SMOTE技术处理不平衡数据...")
    
    # 确保标签是一维的
    if len(labels.shape) > 1:
        labels = labels[:, 0]  # 取第一列作为标签
    
    # 创建SMOTE对象
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    
    # 应用SMOTE
    # 注意：SMOTE需要特征数据，我们使用边的源节点和目标节点作为特征
    X_resampled, y_resampled = smote.fit_resample(edges, labels)
    
    # 将标签转换回二维格式
    y_resampled_2d = np.zeros((len(y_resampled), 2))
    y_resampled_2d[:, 0] = y_resampled
    y_resampled_2d[:, 1] = 1 - y_resampled
    
    print(f"原始数据: {len(edges)} 条边, 正样本比例: {np.sum(labels == 1) / len(labels):.4f}")
    print(f"SMOTE后: {len(X_resampled)} 条边, 正样本比例: {np.sum(y_resampled == 1) / len(y_resampled):.4f}")
    
    return X_resampled, y_resampled_2d
