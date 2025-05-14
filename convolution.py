import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_mean, scatter_max
from robust_aggregation import ro_coefficient
import torch.nn as nn


def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


class Convolution(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_labels,
                 robust_aggr=False,  # whether to use robust aggregation
                 norm_embed=False,
                 bias=True,
                 use_adaptive_agg=True,  # 控制是否使用自适应聚合
                 use_rnn_attention=False,  # 新增：是否使用RNN注意力机制
                 rnn_type='gru',  # 新增：RNN类型
                 num_rnn_layers=1,  # 新增：RNN层数
                 bidirectional=True,  # 新增：是否双向RNN
                 n_heads=4,  # 新增：注意力头数量
                 attn_drop=0.1):  # 新增：注意力dropout率
        super(Convolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.robust_aggr = robust_aggr
        self.norm_embed = norm_embed
        self.num_labels = num_labels
        self.use_adaptive_agg = use_adaptive_agg
        self.use_rnn_attention = use_rnn_attention  # 新增
        
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.trans_weight = Parameter(torch.Tensor(self.num_labels, int(self.in_channels / 4)))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        size = self.weight.size(0)
        size2 = self.trans_weight.size(0)

        uniform(size, self.weight)
        uniform(size, self.bias)
        uniform(size2, self.trans_weight)

    def __repr__(self):
        """
        Create formal string representation.
        """
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)
        
    def get_parameters(self):
        """
        获取模型参数用于正则化
        :return: 模型的所有参数列表
        """
        params = []
        if hasattr(self, 'weight'):
            params.append(self.weight)
        if hasattr(self, 'bias') and self.bias is not None:
            params.append(self.bias)
        if hasattr(self, 'trans_weight'):
            params.append(self.trans_weight)
        if self.use_rnn_attention:
            params.extend(self.rnn_attention.get_parameters())
        return params

#空间聚合层
class ConvolutionBase_in_out(Convolution):
    """
    First layer of TrustGuard with adaptive aggregation.
    """
    def __init__(self, in_channels, out_channels, num_labels, robust_aggr=False, norm_embed=False, bias=True, use_adaptive_agg=True):
        super(ConvolutionBase_in_out, self).__init__(in_channels, out_channels, num_labels, robust_aggr, norm_embed, bias, use_adaptive_agg)
        
        # 自适应聚合参数 - 修正维度
        # 对于ConvolutionBase_in_out，节点特征初始维度预估
        node_feature_dim = int(in_channels / 4)  # 最佳估计
        self.attention_weight = Parameter(torch.Tensor(node_feature_dim, 1))
        self.edge_attention_weight = Parameter(torch.Tensor(int(in_channels / 4), 1))
        
        # 聚合方式权重
        self.agg_weights = Parameter(torch.Tensor(3))  # mean, max, sum
        
        # 初始化
        self.reset_additional_parameters()
        
    def reset_additional_parameters(self):
        """初始化额外参数"""
        nn.init.xavier_uniform_(self.attention_weight)
        nn.init.xavier_uniform_(self.edge_attention_weight)
        nn.init.constant_(self.agg_weights, 1.0/3)  # 平均初始化权重
        
    def get_parameters(self):
        """
        获取所有参数用于正则化，包括自适应聚合参数
        :return: 参数列表
        """
        params = super().get_parameters()
        params.append(self.attention_weight)
        params.append(self.edge_attention_weight)
        params.append(self.agg_weights)
        return params
        
    def compute_attention_weights(self, x, edge_index, edge_features):
        """计算注意力权重"""
        row, col = edge_index
        
        # 基于节点特征的注意力
        source_node = x[row]
        target_node = x[col]
        node_features = source_node * target_node  # 节点特征交互
        
        # 检查注意力权重矩阵的维度是否与节点特征匹配
        if self.attention_weight.size(0) != node_features.size(1):
            # 动态调整注意力权重的维度
            self.attention_weight = Parameter(torch.Tensor(node_features.size(1), 1).to(node_features.device))
            # 重新初始化
            nn.init.xavier_uniform_(self.attention_weight)
        
        node_attention = F.leaky_relu(torch.matmul(node_features, self.attention_weight))
        
        # 基于边特征的注意力
        # 同样检查边特征的维度
        if self.edge_attention_weight.size(0) != edge_features.size(1):
            self.edge_attention_weight = Parameter(torch.Tensor(edge_features.size(1), 1).to(edge_features.device))
            nn.init.xavier_uniform_(self.edge_attention_weight)
            
        edge_attention = F.leaky_relu(torch.matmul(edge_features, self.edge_attention_weight))
        
        # 结合两种注意力
        attention = node_attention + edge_attention
        attention_weights = torch.sigmoid(attention)
        
        return attention_weights
        
    def adaptive_aggregate(self, x, edge_index, edge_features, attention_weights):
        """自适应聚合"""
        row, col = edge_index
        
        # 加权特征
        weighted_edge_features = edge_features * attention_weights
        weighted_node_features = x[col] * attention_weights
        
        # 不同聚合方式的权重
        agg_weights = F.softmax(self.agg_weights, dim=0)
        
        # 出度聚合 - 边特征
        mean_opinion = scatter_mean(weighted_edge_features, row, dim=0, dim_size=x.size(0))
        max_opinion = scatter_max(weighted_edge_features, row, dim=0, dim_size=x.size(0))[0]
        sum_opinion = scatter_add(weighted_edge_features, row, dim=0, dim_size=x.size(0))
        opinion = mean_opinion * agg_weights[0] + max_opinion * agg_weights[1] + sum_opinion * agg_weights[2]
        
        # 出度聚合 - 节点特征
        mean_out = scatter_mean(weighted_node_features, row, dim=0, dim_size=x.size(0))
        max_out = scatter_max(weighted_node_features, row, dim=0, dim_size=x.size(0))[0]
        sum_out = scatter_add(weighted_node_features, row, dim=0, dim_size=x.size(0))
        out = mean_out * agg_weights[0] + max_out * agg_weights[1] + sum_out * agg_weights[2]
        
        # 入度聚合 - 边特征
        mean_inn_opinion = scatter_mean(weighted_edge_features, col, dim=0, dim_size=x.size(0))
        max_inn_opinion = scatter_max(weighted_edge_features, col, dim=0, dim_size=x.size(0))[0]
        sum_inn_opinion = scatter_add(weighted_edge_features, col, dim=0, dim_size=x.size(0))
        inn_opinion = mean_inn_opinion * agg_weights[0] + max_inn_opinion * agg_weights[1] + sum_inn_opinion * agg_weights[2]
        
        # 入度聚合 - 节点特征
        weighted_source = x[row] * attention_weights
        mean_inn = scatter_mean(weighted_source, col, dim=0, dim_size=x.size(0))
        max_inn = scatter_max(weighted_source, col, dim=0, dim_size=x.size(0))[0]
        sum_inn = scatter_add(weighted_source, col, dim=0, dim_size=x.size(0))
        inn = mean_inn * agg_weights[0] + max_inn * agg_weights[1] + sum_inn * agg_weights[2]
        
        return out, opinion, inn, inn_opinion
    
    def forward(self, x, edge_index, edge_label):
        """
        Forward propagation pass with features an indices.
        :param x: node feature matrix.
        :param edge_index: Indices.
        :param edge_label: Edge attribute (i.e., trust relationship) vector.
        """
        row, col = edge_index  # row: trustor index  col: trustee index
        edge_label_trans = torch.matmul(edge_label, self.trans_weight)
        
        # 新增：如果使用RNN注意力机制，先处理时序特征
        if self.use_rnn_attention:
            # 将节点特征重塑为时序形式 [N, T, F]
            # 这里假设每个节点有10个时间步的特征
            batch_size = x.size(0)
            x = x.view(batch_size, -1, self.in_channels)
            # 应用RNN注意力机制
            x = self.rnn_attention(x)
            # 重塑回原始形状
            x = x.view(batch_size, -1)
        
        if not self.robust_aggr:
            if self.use_adaptive_agg:
                # 计算注意力权重
                attention_weights = self.compute_attention_weights(x, edge_index, edge_label_trans)
                
                # 自适应聚合
                out, opinion, inn, inn_opinion = self.adaptive_aggregate(x, edge_index, edge_label_trans, attention_weights)
            else:
                # 使用传统聚合方式
                opinion = scatter_mean(edge_label_trans, row, dim=0, dim_size=x.size(0))
                out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
                inn_opinion = scatter_mean(edge_label_trans, col, dim=0, dim_size=x.size(0))
                inn = scatter_mean(x[row], col, dim=0, dim_size=x.size(0))
        else:
            print('---------- Basic robust aggregation starts! ----------')
            row_out, col_out = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
            ro_out = ro_coefficient(x, 1.4, row_out, col_out)
            
            if self.use_adaptive_agg:
                # 结合鲁棒性系数和注意力权重
                attention_weights = self.compute_attention_weights(x, edge_index, edge_label_trans)
                combined_weights = attention_weights * ro_out
                
                edge_label_trans_out = torch.mul(edge_label_trans, combined_weights)
                x_out = torch.mul(x[col], combined_weights)
            else:
                # 使用传统的鲁棒聚合
                edge_label_trans_out = torch.mul(edge_label_trans, ro_out)
                x_out = torch.mul(x[col], ro_out)
                
            opinion = scatter_add(edge_label_trans_out, row, dim=0, dim_size=x.size(0))
            out = scatter_add(x_out, row, dim=0, dim_size=x.size(0))

            row_in, col_in = edge_index[1].cpu().data.numpy()[:], edge_index[0].cpu().data.numpy()[:]
            ro_in = ro_coefficient(x, 1.4, row_in, col_in)
            
            if self.use_adaptive_agg:
                combined_weights_in = attention_weights * ro_in
                edge_label_trans_in = torch.mul(edge_label_trans, combined_weights_in)
                x_in = torch.mul(x[row], combined_weights_in)
            else:
                edge_label_trans_in = torch.mul(edge_label_trans, ro_in)
                x_in = torch.mul(x[row], ro_in)
                
            inn_opinion = scatter_add(edge_label_trans_in, col, dim=0, dim_size=x.size(0))
            inn = scatter_add(x_in, col, dim=0, dim_size=x.size(0))

        out = torch.cat((out, opinion, inn, inn_opinion), 1)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:  # False
            out = F.normalize(out, p=2, dim=-1)

        return out


class ConvolutionDeep_in_out(Convolution):
    """
    Deep layers of TrustGuard with adaptive aggregation.
    """
    def __init__(self, in_channels, out_channels, num_labels, robust_aggr=False, norm_embed=False, bias=True, use_adaptive_agg=True):
        super(ConvolutionDeep_in_out, self).__init__(in_channels, out_channels, num_labels, robust_aggr, norm_embed, bias, use_adaptive_agg)
        
        # 自适应聚合参数 - 修正维度
        # 对于ConvolutionDeep_in_out，节点特征维度是in_channels/4
        node_feature_dim = int(in_channels / 4)  # 最佳估计
        self.attention_weight = Parameter(torch.Tensor(node_feature_dim, 1))
        self.edge_attention_weight = Parameter(torch.Tensor(int(in_channels / 4), 1))
        
        # 聚合方式权重
        self.agg_weights = Parameter(torch.Tensor(3))  # mean, max, sum
        
        # 初始化
        self.reset_additional_parameters()
        
    def reset_additional_parameters(self):
        """初始化额外参数"""
        nn.init.xavier_uniform_(self.attention_weight)
        nn.init.xavier_uniform_(self.edge_attention_weight)
        nn.init.constant_(self.agg_weights, 1.0/3)  # 平均初始化权重
        
    def get_parameters(self):
        """
        获取所有参数用于正则化，包括自适应聚合参数
        :return: 参数列表
        """
        params = super().get_parameters()
        params.append(self.attention_weight)
        params.append(self.edge_attention_weight)
        params.append(self.agg_weights)
        return params
        
    def compute_attention_weights(self, x, edge_index, edge_features):
        """计算注意力权重"""
        row, col = edge_index
        
        # 基于节点特征的注意力
        source_node = x[row]
        target_node = x[col]
        node_features = source_node * target_node  # 节点特征交互
        
        # 检查注意力权重矩阵的维度是否与节点特征匹配
        if self.attention_weight.size(0) != node_features.size(1):
            # 动态调整注意力权重的维度
            self.attention_weight = Parameter(torch.Tensor(node_features.size(1), 1).to(node_features.device))
            # 重新初始化
            nn.init.xavier_uniform_(self.attention_weight)
        
        node_attention = F.leaky_relu(torch.matmul(node_features, self.attention_weight))
        
        # 基于边特征的注意力
        # 同样检查边特征的维度
        if self.edge_attention_weight.size(0) != edge_features.size(1):
            self.edge_attention_weight = Parameter(torch.Tensor(edge_features.size(1), 1).to(edge_features.device))
            nn.init.xavier_uniform_(self.edge_attention_weight)
            
        edge_attention = F.leaky_relu(torch.matmul(edge_features, self.edge_attention_weight))
        
        # 结合两种注意力
        attention = node_attention + edge_attention
        attention_weights = torch.sigmoid(attention)
        
        return attention_weights
        
    def adaptive_aggregate(self, x, edge_index, edge_features, attention_weights):
        """自适应聚合"""
        row, col = edge_index
        
        # 加权特征
        weighted_edge_features = edge_features * attention_weights
        weighted_node_features = x[col] * attention_weights
        
        # 不同聚合方式的权重
        agg_weights = F.softmax(self.agg_weights, dim=0)
        
        # 出度聚合 - 边特征
        mean_opinion = scatter_mean(weighted_edge_features, row, dim=0, dim_size=x.size(0))
        max_opinion = scatter_max(weighted_edge_features, row, dim=0, dim_size=x.size(0))[0]
        sum_opinion = scatter_add(weighted_edge_features, row, dim=0, dim_size=x.size(0))
        opinion = mean_opinion * agg_weights[0] + max_opinion * agg_weights[1] + sum_opinion * agg_weights[2]
        
        # 出度聚合 - 节点特征
        mean_out = scatter_mean(weighted_node_features, row, dim=0, dim_size=x.size(0))
        max_out = scatter_max(weighted_node_features, row, dim=0, dim_size=x.size(0))[0]
        sum_out = scatter_add(weighted_node_features, row, dim=0, dim_size=x.size(0))
        out = mean_out * agg_weights[0] + max_out * agg_weights[1] + sum_out * agg_weights[2]
        
        # 入度聚合 - 边特征
        mean_inn_opinion = scatter_mean(weighted_edge_features, col, dim=0, dim_size=x.size(0))
        max_inn_opinion = scatter_max(weighted_edge_features, col, dim=0, dim_size=x.size(0))[0]
        sum_inn_opinion = scatter_add(weighted_edge_features, col, dim=0, dim_size=x.size(0))
        inn_opinion = mean_inn_opinion * agg_weights[0] + max_inn_opinion * agg_weights[1] + sum_inn_opinion * agg_weights[2]
        
        # 入度聚合 - 节点特征
        weighted_source = x[row] * attention_weights
        mean_inn = scatter_mean(weighted_source, col, dim=0, dim_size=x.size(0))
        max_inn = scatter_max(weighted_source, col, dim=0, dim_size=x.size(0))[0]
        sum_inn = scatter_add(weighted_source, col, dim=0, dim_size=x.size(0))
        inn = mean_inn * agg_weights[0] + max_inn * agg_weights[1] + sum_inn * agg_weights[2]
        
        return out, opinion, inn, inn_opinion
    
    def forward(self, x, edge_index, edge_label):
        """
        Forward propagation pass with features an indices.
        :param x: Features from previous layer.
        :param edge_index: Indices.
        :param edge_label: Edge attribute (i.e., trust relationship) vector.
        :return out: Abstract convolved features.
        """
        row, col = edge_index
        edge_label_trans = torch.matmul(edge_label, self.trans_weight)
        
        # 新增：如果使用RNN注意力机制，先处理时序特征
        if self.use_rnn_attention:
            # 将节点特征重塑为时序形式 [N, T, F]
            # 这里假设每个节点有10个时间步的特征
            batch_size = x.size(0)
            x = x.view(batch_size, -1, self.in_channels)
            # 应用RNN注意力机制
            x = self.rnn_attention(x)
            # 重塑回原始形状
            x = x.view(batch_size, -1)
        
        if not self.robust_aggr:
            if self.use_adaptive_agg:
                # 计算注意力权重
                attention_weights = self.compute_attention_weights(x, edge_index, edge_label_trans)
                
                # 自适应聚合
                out, opinion, inn, inn_opinion = self.adaptive_aggregate(x, edge_index, edge_label_trans, attention_weights)
            else:
                # 使用传统聚合方式
                opinion = scatter_mean(edge_label_trans, row, dim=0, dim_size=x.size(0))
                out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
                inn_opinion = scatter_mean(edge_label_trans, col, dim=0, dim_size=x.size(0))
                inn = scatter_mean(x[row], col, dim=0, dim_size=x.size(0))
        else:
            print('---------- Deep robust aggregation starts! ----------')
            row_out, col_out = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]
            ro_out = ro_coefficient(x, 1.4, row_out, col_out)
            
            if self.use_adaptive_agg:
                # 结合鲁棒性系数和注意力权重
                attention_weights = self.compute_attention_weights(x, edge_index, edge_label_trans)
                combined_weights = attention_weights * ro_out
                
                edge_label_trans_out = torch.mul(edge_label_trans, combined_weights)
                x_out = torch.mul(x[col], combined_weights)
            else:
                # 使用传统的鲁棒聚合
                edge_label_trans_out = torch.mul(edge_label_trans, ro_out)
                x_out = torch.mul(x[col], ro_out)
                
            opinion = scatter_add(edge_label_trans_out, row, dim=0, dim_size=x.size(0))
            out = scatter_add(x_out, row, dim=0, dim_size=x.size(0))

            row_in, col_in = edge_index[1].cpu().data.numpy()[:], edge_index[0].cpu().data.numpy()[:]
            ro_in = ro_coefficient(x, 1.4, row_in, col_in)
            
            if self.use_adaptive_agg:
                combined_weights_in = attention_weights * ro_in
                edge_label_trans_in = torch.mul(edge_label_trans, combined_weights_in)
                x_in = torch.mul(x[row], combined_weights_in)
            else:
                edge_label_trans_in = torch.mul(edge_label_trans, ro_in)
                x_in = torch.mul(x[row], ro_in)
                
            inn_opinion = scatter_add(edge_label_trans_in, col, dim=0, dim_size=x.size(0))
            inn = scatter_add(x_in, col, dim=0, dim_size=x.size(0))

        out = torch.cat((out, opinion, inn, inn_opinion), 1)
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)

        return out

#时间卷积层
class AttentionNetwork(nn.Module):
    """
    Position-aware self-attention mechanism.
    """
    def __init__(self, input_dim, n_heads, num_time_slots, attn_drop=0.1, residual=True):
        """
        初始化多头注意力网络
        :param input_dim: 输入特征维度
        :param n_heads: 注意力头数量
        :param num_time_slots: 时间槽数量
        :param attn_drop: 注意力dropout率
        :param residual: 是否使用残差连接
        """
        super(AttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.num_time_slots = num_time_slots
        self.attn_drop = attn_drop
        self.residual = residual
        
        # 计算每个头的维度
        self.head_dim = input_dim // n_heads
        assert self.head_dim * n_heads == input_dim, "输入维度必须能被头数整除"
        
        # 初始化位置嵌入
        self.position_embeddings = nn.Parameter(torch.zeros(num_time_slots, input_dim))
        nn.init.xavier_uniform_(self.position_embeddings)
        
        # 初始化、K、V的权重矩阵
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        
        # 初始化权重
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)

        self.lin = nn.Linear(input_dim, input_dim, bias=False)  # False is better (not test much)
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """
        前向传播，支持动态时间槽数量
        :param inputs: 输入张量 [N,T,F]
        :return: 输出张量 [N,T,F]
        """
        N, T, F_dim = inputs.shape  # 将F重命名为F_dim，避免覆盖torch.nn.functional的F
        
        # 动态调整位置嵌入的大小以匹配当前的时间槽数量
        if T > self.num_time_slots:
            # 如果当前时间槽数量大于初始化时的数量，扩展位置嵌入
            with torch.no_grad():
                # 创建新的位置嵌入
                new_position_embeddings = torch.zeros(T, F_dim).to(inputs.device)
                # 复制现有的位置嵌入
                new_position_embeddings[:self.num_time_slots] = self.position_embeddings
                # 对新增的位置进行初始化
                for i in range(self.num_time_slots, T):
                    # 使用最后一个位置的嵌入加上一些随机噪声
                    new_position_embeddings[i] = self.position_embeddings[-1] + torch.randn_like(self.position_embeddings[-1]) * 0.01
                
                # 更新位置嵌入
                self.position_embeddings = nn.Parameter(new_position_embeddings)
                # 更新时间槽数量
                self.num_time_slots = T
                print(f"位置嵌入已扩展至 {T} 个时间槽")
        
        # 生成位置索引
        position_inputs = torch.arange(T).to(inputs.device)
        
        # 添加位置嵌入
        temporal_inputs = inputs + self.position_embeddings[position_inputs]  # [N,T,F_dim]
        
        # 多头注意力机制
        Q = torch.matmul(temporal_inputs, self.Q_embedding_weights)  # [N,T,F_dim]
        K = torch.matmul(temporal_inputs, self.K_embedding_weights)  # [N,T,F_dim]
        V = torch.matmul(temporal_inputs, self.V_embedding_weights)  # [N,T,F_dim]
        
        # 分割多头
        Q_ = torch.cat(torch.split(Q, self.head_dim, dim=2), dim=0)  # [N*h,T,F_dim/h]
        K_ = torch.cat(torch.split(K, self.head_dim, dim=2), dim=0)  # [N*h,T,F_dim/h]
        V_ = torch.cat(torch.split(V, self.head_dim, dim=2), dim=0)  # [N*h,T,F_dim/h]
        
        # 计算注意力分数
        outputs = torch.matmul(Q_, K_.transpose(1, 2))  # [N*h,T,T]
        outputs = outputs / (self.head_dim ** 0.5)
        
        # 应用注意力掩码（可选）
        # 这里可以添加掩码逻辑，例如只关注过去的时间点
        
        # Softmax归一化
        attn_scores = torch.nn.functional.softmax(outputs, dim=2)  # [N*h,T,T]
        
        # Dropout
        if self.training and self.attn_drop > 0:
            attn_scores = torch.nn.functional.dropout(attn_scores, p=self.attn_drop)
        
        # 加权求和
        outputs = torch.matmul(attn_scores, V_)  # [N*h,T,F_dim/h]
        
        # 合并多头
        outputs = torch.cat(torch.split(outputs, N, dim=0), dim=2)  # [N,T,F_dim]
        
        
        # 残差连接
        if self.residual:
            outputs = outputs + inputs
        
        # 返回结果
        return outputs
        


    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        # outputs = F.elu(self.lin(inputs))  # elu is worse than relu
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
        
    def get_parameters(self):
        """
        获取所有参数用于正则化
        :return: 参数列表
        """
        params = []
        params.append(self.position_embeddings)
        params.append(self.Q_embedding_weights)
        params.append(self.K_embedding_weights)
        params.append(self.V_embedding_weights)
        # 包含线性层参数
        for param in self.lin.parameters():
            params.append(param)
        return params


class GraphConvolutionalNetwork(nn.Module):
    """
    Graph convolutional network class with adaptive aggregation.
    """
    def __init__(self, device, args, X, num_labels):
        super(GraphConvolutionalNetwork, self).__init__()
        self.args = args
        torch.manual_seed(self.args.seed)  # fixed seed == 42
        self.device = device
        self.X = X
        self.dropout = self.args.dropout
        self.num_labels = num_labels
        self.use_adaptive_agg = True if (hasattr(args, 'use_adaptive_agg') and args.use_adaptive_agg) else True  # 默认使用自适应聚合
        self.setup_layers()

    def setup_layers(self):
        """
        Constructing trust propagation layers with adaptive aggregation.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers  # layers = [32,64,32]
        self.layers = len(self.neurons)
        self.aggregators = []

        # 使用自适应聚合创建基础层
        self.base_aggregator = ConvolutionBase_in_out(
            self.X.shape[1] * 4, 
            self.neurons[0], 
            self.num_labels,
            robust_aggr=self.args.robust_aggr if hasattr(self.args, 'robust_aggr') else False,
            use_adaptive_agg=self.use_adaptive_agg
        ).to(self.device)
        
        # 使用自适应聚合创建深层
        for i in range(1, self.layers):
            self.aggregators.append(
                ConvolutionDeep_in_out(
                    self.neurons[i - 1] * 4, 
                    self.neurons[i], 
                    self.num_labels,
                    robust_aggr=self.args.robust_aggr if hasattr(self.args, 'robust_aggr') else False,
                    use_adaptive_agg=self.use_adaptive_agg
                ).to(self.device))

        self.aggregators = ListModule(*self.aggregators)

    def forward(self, train_edges, y_train):
        """
        Trust propagation and aggregation.
        return output: node embeddings at the last layer.
        """
        h = []
        self.X = F.dropout(self.X, self.dropout, training=self.training)
        h.append(torch.relu(self.base_aggregator(self.X, train_edges, y_train)))

        for i in range(1, self.layers):
            h[-1] = F.dropout(h[-1], self.dropout, training=self.training)
            h.append(torch.relu(self.aggregators[i - 1](h[i - 1], train_edges, y_train)))
        output = h[-1]

        return output
        
    def get_all_parameters(self):
        """
        获取网络中的所有参数用于正则化
        :return: 所有参数列表
        """
        all_params = []
        # 添加基础聚合器参数
        all_params.extend(self.base_aggregator.get_parameters())
        # 添加深层聚合器参数
        for i in range(len(self.aggregators)):
            all_params.extend(self.aggregators[i].get_parameters())
        return all_params
    
    def calculate_regularization(self, regularization_type="L2", l1_lambda=1e-5, l2_lambda=1e-4):
        """
        计算模型的正则化项
        :param regularization_type: 正则化类型，可选 'L1', 'L2' 或 'none'
        :param l1_lambda: L1正则化系数
        :param l2_lambda: L2正则化系数
        :return: 正则化损失
        """
        if regularization_type.lower() == "none":
            return 0.0
            
        all_params = self.get_all_parameters()
        reg_loss = 0.0
        
        if regularization_type.upper() == "L1":
            # L1正则化（Lasso）
            for param in all_params:
                reg_loss += l1_lambda * torch.sum(torch.abs(param))
        elif regularization_type.upper() == "L2":
            # L2正则化（Ridge）
            for param in all_params:
                reg_loss += l2_lambda * torch.sum(param ** 2)
        
        return reg_loss

