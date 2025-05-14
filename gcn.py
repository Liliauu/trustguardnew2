import time
import torch
import numpy as np
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from utils import calculate_auc, setup_features, precision_recall_curve
from convolution import GraphConvolutionalNetwork, AttentionNetwork
from dataset import get_snapshot_index
import copy
import torch.nn as nn
import matplotlib.pyplot as plt


class RegularizationConfig:
    def __init__(self):
        self.l1_lambda = 1e-5
        self.l2_lambda = 1e-4
        self.elastic_net_ratio = 0.5  # 弹性网络混合比例
        self.adaptive_lambda = True   # 是否自适应调整正则化系数


class TrustGuard(torch.nn.Module):
    """
    Graph convolutional network class for TrustGuard.
    """
    def __init__(self, device, args, X, num_labels):
        """
        Initialize TrustGuard model.
        :param device: GPU device.
        :param args: Arguments for TrustGuard model.
        :param X: Input features.
        :param num_labels: Number of labels.
        """
        super(TrustGuard, self).__init__()
        self.args = args
        torch.manual_seed(self.args.seed)  # fixed seed == 42
        self.device = device
        self.X = X
        #设置 dropout rate，即在训练过程中，每次迭代时，每个神经元有多少概率被随机失活，确保结果的可重复性
        self.dropout = self.args.dropout
        self.num_labels = num_labels
        self.build_model()
        self.regression_weights = Parameter(torch.Tensor(self.args.layers[-1]*2, self.num_labels))
        init.xavier_normal_(self.regression_weights)  # initialize regression_weights
        self.logs = {
            "training_time": [],
            "incremental_update_time": [],  # 新增记录增量更新时间的日志
            "distillation_loss": [],  # 记录蒸馏损失
        }
        self.teacher_model = None  # 初始化教师模型为None

    def build_model(self):
        """
        Constructing spatial and temporal layers.
        """
        self.structural_layer = GraphConvolutionalNetwork(self.device, self.args, self.X, self.num_labels)
        self.temporl_layer = AttentionNetwork(input_dim=self.args.layers[-1],n_heads=self.args.attention_head,num_time_slots=self.args.train_time_slots,attn_drop=0.5,residual=True)

    def calculate_loss_function(self, z, train_edges, target, distillation=False, teacher_logits=None):
        """
        计算损失函数，支持知识蒸馏
        :param z: 节点嵌入
        :param train_edges: 训练边
        :param target: 目标标签
        :param distillation: 是否使用知识蒸馏
        :param teacher_logits: 教师模型的输出 logits
        :return loss: 损失值
        """
        start_node, end_node = z[train_edges[0], :], z[train_edges[1], :]

        features = torch.cat((start_node, end_node), 1)
        predictions = torch.mm(features, self.regression_weights)

        # 处理不平衡数据
        class_weight = torch.FloatTensor(1 / np.bincount(target.cpu()) * features.size(0))
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(self.device)
        loss_term = criterion(predictions, target)
        
        # 知识蒸馏损失
        distill_loss = 0.0
        if distillation and teacher_logits is not None:
            # 温度参数
            temperature = 2.0
            # 软化logits
            soft_targets = F.softmax(teacher_logits / temperature, dim=1)
            soft_prob = F.log_softmax(predictions / temperature, dim=1)
            # KL散度损失
            distill_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
            # 记录蒸馏损失
            self.logs["distillation_loss"].append(distill_loss.item())
        
        # 添加正则化项
        l1_lambda = getattr(self.args, 'l1_lambda', 1e-5)  # L1正则化系数
        l2_lambda = getattr(self.args, 'l2_lambda', 1e-4)  # L2正则化系数
        reg_type = getattr(self.args, 'regularization_type', 'L2').upper()
        
        reg_loss = 0.0
        # 收集需要正则化的参数
        weight_params = [
            self.regression_weights,  # 回归权重
            self.structural_layer.base_aggregator.weight,  # 基础聚合层权重
        ]
        
        # 添加卷积层权重
        for aggregator in self.structural_layer.aggregators:
            weight_params.append(aggregator.weight)
        
        # 添加时序层权重
        weight_params.extend([
            self.temporl_layer.Q_embedding_weights,
            self.temporl_layer.K_embedding_weights,
            self.temporl_layer.V_embedding_weights
        ])
        
        # 根据正则化类型应用不同的正则化
        if reg_type == 'L1' and l1_lambda > 0:
            for param in weight_params:
                reg_loss += l1_lambda * torch.sum(torch.abs(param))
        elif reg_type == 'L2' and l2_lambda > 0:
            for param in weight_params:
                reg_loss += l2_lambda * torch.sum(param ** 2)
        elif reg_type == 'ELASTIC_NET':
            for param in weight_params:
                l1_term = l1_lambda * torch.sum(torch.abs(param))
                l2_term = l2_lambda * torch.sum(param ** 2)
                elastic_ratio = getattr(self.args, 'elastic_net_ratio', 0.5)
                reg_loss += elastic_ratio * l1_term + (1 - elastic_ratio) * l2_term
        
        # 结合原始损失、蒸馏损失和正则化损失
        # 蒸馏权重，平衡标签损失和蒸馏损失
        alpha = 0.5 if distillation else 0.0
        total_loss = (1 - alpha) * loss_term + alpha * distill_loss + reg_loss
        
        return total_loss, predictions

    
    def forward(self, train_edges, y, y_train, index_list, training=True, distillation=False, teacher_model=None):
        """
        前向传播，支持知识蒸馏
        :param train_edges: 训练边
        :param y: 目标标签
        :param y_train: 训练标签
        :param index_list: 快照索引列表
        :param training: 是否为训练模式
        :param distillation: 是否使用知识蒸馏
        :param teacher_model: 教师模型
        :return: 损失、输出嵌入和预测结果
        """
        structural_out = []
        index0 = 0
        for i in range(self.args.train_time_slots):
            structural_out.append(self.structural_layer(train_edges[:, index0:index_list[i]], y_train[index0:index_list[i], :]))
            index0 = index_list[i]

        structural_out = torch.stack(structural_out)
        structural_out = structural_out.permute(1,0,2)  # [N,T,F] [5881,7,32]
        temporal_all = self.temporl_layer(structural_out)  # [N,T,F]
        temporal_out = temporal_all[:, self.args.train_time_slots-1, :].squeeze()  # [N,F]

        loss = None
        predictions = None
        if training and y is not None:
            # 获取教师模型的预测结果
            teacher_logits = None
            if distillation and teacher_model is not None:
                with torch.no_grad():
                    _, teacher_out, _ = teacher_model(train_edges, y, y_train, index_list, training=False)
                    start_node, end_node = teacher_out[train_edges[0], :], teacher_out[train_edges[1], :]
                    teacher_features = torch.cat((start_node, end_node), 1)
                    teacher_logits = torch.mm(teacher_features, teacher_model.regression_weights)
            
            loss, predictions = self.calculate_loss_function(temporal_out, train_edges, y, distillation, teacher_logits)

        return loss, temporal_out, predictions

    
        
    def _evaluate_embeddings(self, embeddings, edges, labels):
        """
        评估嵌入向量的性能指标
        
        :param embeddings: 模型生成的节点嵌入向量
        :param edges: 测试边
        :param labels: 测试标签
        :return: 性能指标字典
        """
        # 使用嵌入向量计算边的预测分数
        score_edges = torch.from_numpy(np.array(self.obs, dtype=np.int64).T).type(torch.long).to(self.device)
        test_z = torch.cat((embeddings[score_edges[0, :], :], embeddings[score_edges[1, :], :]), 1)
        
        # 计算分数和预测
        if isinstance(self, TrustGuard):
            scores = torch.mm(test_z, self.regression_weights.to(self.device))
        else:
            scores = torch.mm(test_z, self.model.regression_weights.to(self.device))
            
        predictions = F.softmax(scores, dim=1)
        
        # 计算各种性能指标
        mcc, auc, acc_balanced, precision, f1_micro, f1_macro = calculate_auc(predictions, self.y_test_obs)
        
        # 返回结果字典
        return {
            "MCC": mcc,
            "AUC": auc,
            "ACC_Balanced": acc_balanced,
            "Precision": precision,
            "F1_Micro": f1_micro,
            "F1_Macro": f1_macro
        }

    def sample_nodes(self, edge_batch, sample_size):
        """
        对边批次中的节点进行采样
        """
        nodes = torch.unique(edge_batch.flatten())
        if len(nodes) > sample_size:
            idx = torch.randperm(len(nodes))[:sample_size]
            sampled_nodes = nodes[idx]
            return sampled_nodes
        return nodes

    def sample_neighbors(self, nodes, edge_index, num_neighbors):
        """
        对节点的邻居进行采样
        """
        row, col = edge_index
        node_mask = torch.isin(row, nodes)
        sampled_edges = edge_index[:, node_mask]
        
        # 对每个节点采样固定数量的邻居
        unique_nodes = torch.unique(sampled_edges[0])
        sampled_neighbors = []
        
        for node in unique_nodes:
            neighbors = col[row == node]
            if len(neighbors) > num_neighbors:
                idx = torch.randperm(len(neighbors))[:num_neighbors]
                sampled_neighbors.append(neighbors[idx])
            else:
                sampled_neighbors.append(neighbors)
            
        return torch.cat(sampled_neighbors)

    def calculate_regularization(self, regularization_type="L2", config=None):
        if regularization_type.upper() == "ELASTIC_NET":
            # 弹性网络正则化
            l1_loss = sum(torch.abs(p) for p in self.parameters())
            l2_loss = sum(p.pow(2) for p in self.parameters())
            return config.elastic_net_ratio * config.l1_lambda * l1_loss + \
                   (1 - config.elastic_net_ratio) * config.l2_lambda * l2_loss

    def adaptive_regularization(self, model, validation_loss):
        # 根据验证损失动态调整正则化强度
        if validation_loss > self.best_val_loss:
            self.l1_lambda *= 1.1
            self.l2_lambda *= 1.1
        else:
            self.l1_lambda *= 0.9
            self.l2_lambda *= 0.9




class GCNTrainer(object):
    """
    Object to train and score the TrustGuard, log the model behaviour and save the output.
    """
    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure.
        """
        self.args = args
        self.edges = edges
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_start_time = time.time()
        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary for recording performance.
        """
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["performance"] = [["Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro"]]
        self.logs["training_time"] = [["Epoch", "Seconds"]]
        self.logs["incremental_update_time"] = []  # 新增记录增量更新时间的日志

    def setup_dataset(self):
        """
        创建训练快照和测试快照。
        该函数根据指定的时间槽和数据路径获取快照索引列表，分割训练集和测试集，
        并为单时间槽预测或多时间槽预测准备数据。此外，它还会统计信任和不信任分布，
        并将数据转换为PyTorch张量格式以供后续处理。
        """
        
        self.index_list = get_snapshot_index(self.args.time_slots, data_path=self.args.data_path)
        index_t = self.index_list[self.args.train_time_slots-1]

        print("--------------- Getting training and testing snapshots starts ---------------")
        print('Snapshot index',self.index_list,index_t)  # index_t denotes the index of snapshot t
        self.train_edges = self.edges['edges'][:index_t]
        self.y_train = self.edges['labels'][:index_t]
        train_set = set(list(self.train_edges.flatten()))

        # single-timeslot prediction
        if self.args.single_prediction:
            # index_t_1 = self.index_list[self.args.train_time_slots-2]  # index of snapshot t-1, used for single-timeslot prediction on unobserved nodes, i.e., task 3
            # train_pre = set(list(self.train_edges[:index_t_1].flatten()))  # for task 3
            # train_t = set(list(self.train_edges[index_t_1:].flatten()))  # for task 3

            index_t1 = self.index_list[self.args.train_time_slots]  # index of snapshot t+1
            self.test_edges = self.edges['edges'][index_t:index_t1]
            self.y_test = self.edges['labels'][index_t:index_t1]
            print('{} edges at snapshot t+1'.format(len(self.test_edges)))

            self.obs = []  # observed nodes' edges
            self.y_test_obs = []
            # self.unobs = []  # for task 3
            # self.y_test_unobs = []  # for task 3
            for i in range(len(self.test_edges)):
                tr = self.test_edges[i][0]
                te = self.test_edges[i][1]
                if tr in train_set and te in train_set:
                    self.obs.append(self.test_edges[i])
                    self.y_test_obs.append(self.y_test[i])
                # for task 3
                # if tr in train_t and tr not in train_pre and te not in train_t and te in train_pre:
                #     self.unobs.append(self.test_edges[i])
                #     self.y_test_unobs.append(self.y_test[i])
                # elif te in train_t and te not in train_pre and tr not in train_t and tr in train_pre:
                #     self.unobs.append(self.test_edges[i])
                #     self.y_test_unobs.append(self.y_test[i])
                # elif tr in train_t and te in train_t and tr not in train_pre and te not in train_pre:
                #     self.unobs.append(self.test_edges[i])
                #     self.y_test_unobs.append(self.y_test[i])

            self.pos_count = 0
            self.neg_count = 0
            for i in range(len(self.y_test_obs)):
                if self.y_test_obs[i][0] == 1:
                    self.pos_count += 1
                else:
                    self.neg_count += 1
            print('Trust and distrust distribution:',self.pos_count,self.neg_count)
            # print('Observed single-timeslot test edges'.format(len(self.unobs)))  # for task 3
            print('Observed single-timeslot test edges:',len(self.obs))

            self.obs = np.array(self.obs)
            self.y_test_obs = np.array(self.y_test_obs)
            # self.unobs = np.array(self.unobs)  # for task 3
            # self.y_test_unobs = np.array(self.y_test_unobs)  # for task 3
        else:   # Multi-timeslot prediction, predict latter three snapshots, i.e., task 2
            index_pre = self.index_list[self.args.train_time_slots - 1]
            index_lat = self.index_list[self.args.train_time_slots + 2]
            self.test_edges = self.edges['edges'][index_pre:index_lat]
            self.y_test = self.edges['labels'][index_pre:index_lat]
            print('{} edges from snapshot t+1 to snapshot t+3'.format(len(self.test_edges)))

            self.obs = []  # observed nodes' edges
            self.y_test_obs = []

            for i in range(len(self.test_edges)):
                if self.test_edges[i][0] in train_set and self.test_edges[i][1] in train_set:
                    self.obs.append(self.test_edges[i])
                    self.y_test_obs.append(self.y_test[i])

            self.pos_count = 0
            self.neg_count = 0
            for i in range(len(self.y_test_obs)):
                if self.y_test_obs[i][0] == 1:
                    self.pos_count += 1
                else:
                    self.neg_count += 1
            print('Trust and distrust distribution:',self.pos_count,self.neg_count)
            print('Observed multi-timeslot test edges:',len(self.obs))

            self.obs = np.array(self.obs)
            self.y_test_obs = np.array(self.y_test_obs)
        print("--------------- Getting training and testing snapshots ends ---------------")

        self.X = setup_features(self.args)  # Setting up the node features as a numpy array.
        self.num_labels = np.shape(self.y_train)[1]

        self.y = torch.from_numpy(self.y_train[:,1]).type(torch.long).to(self.device)
        # convert vector to number 0/1, 0 represents trust and 1 represents distrust

        self.train_edges = torch.from_numpy(np.array(self.train_edges, dtype=np.int64).T).type(torch.long).to(self.device)  # (2, #edges)
        self.y_train = torch.from_numpy(np.array(self.y_train, dtype=np.float32)).type(torch.float).to(self.device)
        self.num_labels = torch.from_numpy(np.array(self.num_labels, dtype=np.int64)).type(torch.long).to(self.device)
        self.X = torch.from_numpy(self.X).to(self.device)

        # 添加节点特征工程
        self.add_node_features()

#以下都是全新修改，添加的模块
    def add_node_features(self):
        """添加额外的节点特征以提高精确率"""
        # 计算每个节点的度
        node_degrees = torch.zeros(self.X.shape[0], dtype=torch.float).to(self.device)
        for i in range(self.train_edges.shape[1]):
            node_degrees[self.train_edges[0, i]] += 1
            node_degrees[self.train_edges[1, i]] += 1
        
        # 归一化度特征
        max_degree = torch.max(node_degrees)
        node_degrees = node_degrees / max_degree
        
        # 将度特征添加到节点特征中
        degree_features = node_degrees.view(-1, 1)
        self.X = torch.cat([self.X, degree_features], dim=1)
        
        print(f"添加了节点度特征，特征维度从 {self.X.shape[1]-1} 增加到 {self.X.shape[1]}")

    def create_batches(self, train_edges, y_train, batch_size):
        """
        创建边的批次
        """
        num_edges = train_edges.size(1)
        indices = torch.randperm(num_edges)
        batches = []
        
        for i in range(0, num_edges, batch_size):
            batch_indices = indices[i:min(i + batch_size, num_edges)]
            edge_batch = train_edges[:, batch_indices]
            label_batch = y_train[batch_indices]
            batches.append((edge_batch, label_batch))
        
        return batches

    def create_and_train_model(self, new_edge_path=None, is_incremental=False):
        """
        使用批处理进行模型训练，支持增量更新
        
        :param new_edge_path: 新边数据的文件路径，用于增量更新
        :param is_incremental: 是否为增量更新模式
        """
        # 如果是增量更新模式，先处理新数据
        if is_incremental and new_edge_path:
            print("\n--------------- 开始增量更新 ---------------")
            start_time = time.time()
            
            # 1. 保存当前模型作为教师模型
            self.model.eval()
            teacher_model = copy.deepcopy(self.model)
            teacher_model.eval()
            
            # 2. 加载并处理新数据
            data_start_time = time.time()
            new_edges, new_labels = self.process_incremental_data(new_edge_path)
            print(f"加载了 {len(new_edges)} 条新边数据")
            
            # 3. 将新数据与旧数据合并
            self.merge_incremental_data(new_edges, new_labels)
            data_process_time = time.time() - data_start_time
            
            # 4. 检查并更新时序层的时间槽数量
            if hasattr(self.model, 'temporl_layer') and hasattr(self.model.temporl_layer, 'num_time_slots'):
                current_slots = self.args.train_time_slots
                if current_slots > self.model.temporl_layer.num_time_slots:
                    print(f"更新时序层时间槽: {self.model.temporl_layer.num_time_slots} -> {current_slots}")
                    # 创建新的位置嵌入
                    old_embeddings = self.model.temporl_layer.position_embeddings.data
                    input_dim = old_embeddings.size(1)
                    new_embeddings = torch.zeros(current_slots, input_dim).to(self.device)
                    # 复制现有的位置嵌入
                    new_embeddings[:self.model.temporl_layer.num_time_slots] = old_embeddings
                    # 对新增的位置进行初始化
                    for i in range(self.model.temporl_layer.num_time_slots, current_slots):
                        new_embeddings[i] = old_embeddings[-1] + torch.randn_like(old_embeddings[-1]) * 0.01
                    
                    # 更新位置嵌入
                    self.model.temporl_layer.position_embeddings = nn.Parameter(new_embeddings)
                    self.model.temporl_layer.num_time_slots = current_slots
            
            # 5. 设置增量训练参数
            self.model.train()
            # 使用较小的学习率进行微调
            incremental_lr = self.args.learning_rate * 0.005
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=incremental_lr,
                weight_decay=self.args.weight_decay
            )
            
            # 添加loss监控
            best_loss = float('inf')
            patience = 5
            patience_counter = 0
            
            # 6. 使用知识蒸馏进行增量更新训练
            training_start_time = time.time()
            increment_epochs = trange(self.args.epochs, desc="Incremental Loss")
            for epoch in increment_epochs:
                epoch_start_time = time.time()
                optimizer.zero_grad()
                
                # 使用知识蒸馏
                loss, temporal_out, predictions = self.model(
                    self.train_edges, 
                    self.y, 
                    self.y_train, 
                    self.index_list, 
                    training=True,
                    distillation=True,
                    teacher_model=teacher_model
                )
                
                loss.backward()
                increment_epochs.set_description(f"Incremental Update (Loss={round(loss.item(), 4)})")
                optimizer.step()
                
                # 每轮都评估一次模型
                if (epoch + 1) % 5 == 0 or epoch == self.args.epochs - 1:
                    metrics = self.score_model(epoch, prefix="增量更新")
                    current_loss = loss.item()
                    
                    # 早停策略：如果loss不再下降，提前停止
                    if current_loss < best_loss:
                        best_loss = current_loss
                        patience_counter = 0
                        # 保存最佳模型
                        best_model = copy.deepcopy(self.model)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Loss不再下降，在第{epoch+1}轮提前停止训练")
                            # 恢复最佳模型
                            self.model = best_model
                            break
                
                epoch_time = time.time() - epoch_start_time
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1} 耗时: {epoch_time:.2f} 秒")
            
            training_time = time.time() - training_start_time
            print(f"训练总耗时: {training_time:.2f} 秒")
            
            # 7. 记录增量更新的时间
            update_time = time.time() - start_time
            self.logs["incremental_update_time"].append(update_time)
            
            print(f"增量更新完成，总耗时: {update_time:.2f} 秒")
            print(f"   - 模型训练: {training_time:.2f} 秒")
            print("--------------- 增量更新结束 ---------------\n")
            
            # 8. 返回最终性能指标
            return self.evaluate_incremental_performance()
        
        # 如果不是增量更新模式，执行常规训练
        else:
            # 如果是首次训练，初始化模型
            if not hasattr(self, 'model') or self.model is None:
                self.model = TrustGuard(self.device, self.args, self.X, self.num_labels).to(self.device)
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), 
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay
                )
            
            epochs = trange(self.args.epochs, desc="Loss")
            for epoch in epochs:
                start_time = time.time()
                self.model.train()
                self.optimizer.zero_grad()
                loss, _, _ = self.model(self.train_edges, self.y, self.y_train, self.index_list)
                loss.backward()
                self.optimizer.step()
                epochs.set_description("TrustGuard (Loss=%g)" % round(loss.item(), 4))
                self.score_model(epoch)
                self.logs["training_time"].append([epoch + 1, time.time() - start_time])
            
            return self.score_model(self.args.epochs - 1)

    def score_model(self, epoch, prefix=""):
        """
        评估模型在测试集上的性能
        :param epoch: 当前轮次
        :param prefix: 评估结果的前缀标识
        """
        self.model.eval()
        _, self.train_z, _ = self.model(self.train_edges, self.y, self.y_train, self.index_list, training=False)
        score_edges = torch.from_numpy(np.array(self.obs, dtype=np.int64).T).type(torch.long).to(self.device)
        test_z = torch.cat((self.train_z[score_edges[0, :], :], self.train_z[score_edges[1, :], :]), 1)
        scores = torch.mm(test_z, self.model.regression_weights.to(self.device))
        predictions = F.softmax(scores, dim=1)

        mcc, auc, acc_balanced, precision, f1_micro, f1_macro = calculate_auc(predictions, self.y_test_obs)
        
        # 只在增量更新时输出指标
        if prefix == "增量更新":
            print(f'\n{prefix}评估结果：')
            print('MCC: %.4f' % mcc)
            print('AUC: %.4f' % auc)
            print('ACC_Balanced: %.4f' % acc_balanced)
            print('Precision: %.4f' % precision)
            print('F1_Micro: %.4f' % f1_micro)
            print('F1_Macro: %.4f' % f1_macro)
            print('-' * 50)

        self.logs["performance"].append([epoch + 1, mcc, auc, acc_balanced, precision, f1_micro, f1_macro])

        return {
            "MCC": mcc,
            "AUC": auc,
            "ACC_Balanced": acc_balanced,
            "Precision": precision,
            "F1_Micro": f1_micro,
            "F1_Macro": f1_macro
        }

    def process_incremental_data(self, new_edge_path):
        """
        处理增量边数据
        
        :param new_edge_path: 新边数据的文件路径
        :return: 处理后的新边数据和标签
        """
        from utils import process_incremental_data
        return process_incremental_data(self.args, new_edge_path)

    def merge_incremental_data(self, new_edges, new_labels):
        """
        将新数据与旧数据合并
        
        :param new_edges: 新边数据
        :param new_labels: 新标签数据
        """
        # 将NumPy数组转换为PyTorch张量并移至设备
        new_edges_tensor = torch.from_numpy(np.array(new_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        new_labels_tensor = torch.from_numpy(np.array(new_labels, dtype=np.float32)).type(torch.float).to(self.device)
        new_y = torch.from_numpy(new_labels[:, 1]).type(torch.long).to(self.device)
        
        # 合并边和标签数据
        self.train_edges = torch.cat([self.train_edges, new_edges_tensor], dim=1)
        self.y_train = torch.cat([self.y_train, new_labels_tensor], dim=0)
        self.y = torch.cat([self.y, new_y], dim=0)
        
        # 更新索引列表
        old_last_index = self.index_list[-1]
        new_last_index = old_last_index + len(new_edges)
        self.index_list.append(new_last_index)
        
        # 更新训练时段数
        self.args.train_time_slots += 1
        
        print(f"数据合并完成: 共 {self.train_edges.shape[1]} 条边, {self.args.train_time_slots} 个时段")

    def evaluate_incremental_performance(self):
        """
        评估增量更新后的模型性能
        """
        self.model.eval()
        _, train_z, _ = self.model(self.train_edges, self.y, self.y_train, self.index_list, training=False)
        
        # 在测试集上评估
        score_edges = torch.from_numpy(np.array(self.obs, dtype=np.int64).T).type(torch.long).to(self.device)
        test_z = torch.cat((train_z[score_edges[0, :], :], train_z[score_edges[1, :], :]), 1)
        scores = torch.mm(test_z, self.model.regression_weights.to(self.device))
        predictions_prob = F.softmax(scores, dim=1)
        
        # 确保y_test_obs是一维的
        if isinstance(self.y_test_obs, torch.Tensor) and self.y_test_obs.dim() > 1:
            y_test = self.y_test_obs[:, 1]  # 取第二列作为正类标签
        else:
            y_test = self.y_test_obs
        
        # 计算评估指标
        mcc, auc, acc_balanced, _, f1_micro, f1_macro = calculate_auc(predictions_prob, y_test)
        
        return {
            "MCC": mcc,
            "AUC": auc,
            "ACC_Balanced": acc_balanced,
            "F1_Micro": f1_micro,
            "F1_Macro": f1_macro,
            "Update_Time": self.logs["incremental_update_time"][-1]
        }


class RegularizationMonitor:
    def __init__(self):
        self.reg_loss_history = []
        
    def log_reg_loss(self, loss):
        self.reg_loss_history.append(loss)
        
    def plot_reg_loss(self):
        plt.plot(self.reg_loss_history)
        plt.title('Regularization Loss History')
        plt.show()
